import streamlit as st
import cv2
import numpy as np
from ultralytics import YOLO
import time
import datetime
import sqlite3
import pandas as pd
import plotly.graph_objects as go
from collections import deque
import torch

# Initialize session state for all pages
if 'camera_running' not in st.session_state:
    st.session_state.camera_running = False
if 'object_tracker' not in st.session_state:
    st.session_state.object_tracker = {}
if 'next_object_id' not in st.session_state:
    st.session_state.next_object_id = 0
if 'tracking_history' not in st.session_state:
    st.session_state.tracking_history = deque(maxlen=1000)
if 'time_in_frame' not in st.session_state:
    st.session_state.time_in_frame = {}
if 'paths' not in st.session_state:
    st.session_state.paths = {}
if 'frame_count' not in st.session_state:
    st.session_state.frame_count = 0

# Page selection
page = st.sidebar.selectbox("Choose Page", ["🎥 Live Video", "📊 Dashboard", "📈 Reports & Analytics"])

if page == "🎥 Live Video":
    # --- Professional Branding Header ---
    with st.container():
        col_logo1, col_title, col_logo2 = st.columns([1, 6, 1])
        with col_logo1:
            st.image("branding/sydneyairport.png", width=90)
        with col_title:
            st.markdown("""
                <div style='text-align: center; color: #003366; font-size: 2.2rem; font-weight: bold; letter-spacing: 1px;'>
                    Sydney Airport | Eye4.ai Live Video Feed
                </div>
                <div style='text-align: center; color: #00AEEF; font-size: 1.1rem; font-weight: 500;'>
                    Real-time Multi-Object Detection & Face Blurring
                </div>
            """, unsafe_allow_html=True)
        with col_logo2:
            st.image("branding/eye4ai.png", width=70)

    # --- Custom Theme Styling ---
    st.markdown("""
        <style>
        .stApp {
            background: linear-gradient(120deg, #f0f6fa 0%, #e6f0fa 100%);
        }
        .stButton>button {
            background: #00AEEF;
            color: white;
            border-radius: 6px;
            font-weight: 600;
            border: none;
            padding: 0.5em 1.5em;
            margin: 0.5em 0;
        }
        </style>
    """, unsafe_allow_html=True)

    # Check for GPU
    if torch.cuda.is_available():
        device = 'cuda'
        st.success("🚀 GPU detected and will be used for inference.")
    else:
        device = 'cpu'
        st.warning("⚠️ No GPU detected. Running on CPU.")

    # Load YOLO model (multi-object detection)
    @st.cache_resource
    def load_model():
        model = YOLO('yolov8n.pt')
        model.to(device)
        return model
    
    model = load_model()

    # Load OpenCV face detector
    @st.cache_resource
    def load_face_cascade():
        return cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    
    face_cascade = load_face_cascade()

    # COCO class names for YOLO
    COCO_CLASSES = {
        0: 'person', 1: 'bicycle', 2: 'car', 3: 'motorcycle', 4: 'airplane', 5: 'bus',
        6: 'train', 7: 'truck', 8: 'boat', 9: 'traffic light', 10: 'fire hydrant',
        11: 'stop sign', 12: 'parking meter', 13: 'bench', 14: 'bird', 15: 'cat',
        16: 'dog', 17: 'horse', 18: 'sheep', 19: 'cow', 20: 'elephant', 21: 'bear',
        22: 'zebra', 23: 'giraffe', 24: 'backpack', 25: 'umbrella', 26: 'handbag',
        27: 'tie', 28: 'suitcase', 29: 'frisbee', 30: 'skis', 31: 'snowboard',
        32: 'sports ball', 33: 'kite', 34: 'baseball bat', 35: 'baseball glove',
        36: 'skateboard', 37: 'surfboard', 38: 'tennis racket', 39: 'bottle',
        40: 'wine glass', 41: 'cup', 42: 'fork', 43: 'knife', 44: 'spoon',
        45: 'bowl', 46: 'banana', 47: 'apple', 48: 'sandwich', 49: 'orange',
        50: 'broccoli', 51: 'carrot', 52: 'hot dog', 53: 'pizza', 54: 'donut',
        55: 'cake', 56: 'chair', 57: 'couch', 58: 'potted plant', 59: 'bed',
        60: 'dining table', 61: 'toilet', 62: 'tv', 63: 'laptop', 64: 'mouse',
        65: 'remote', 66: 'keyboard', 67: 'cell phone', 68: 'microwave',
        69: 'oven', 70: 'toaster', 71: 'sink', 72: 'refrigerator', 73: 'book',
        74: 'clock', 75: 'vase', 76: 'scissors', 77: 'teddy bear', 78: 'hair drier',
        79: 'toothbrush'
    }

    # Color palette for different object classes
    CLASS_COLORS = {
        'person': (0, 255, 0),          # Green
        'car': (255, 0, 0),             # Red
        'bicycle': (0, 255, 255),       # Cyan
        'motorcycle': (255, 255, 0),    # Yellow
        'bus': (255, 0, 255),           # Magenta
        'truck': (128, 0, 128),         # Purple
        'chair': (255, 165, 0),         # Orange
        'laptop': (0, 128, 255),        # Blue
        'cell phone': (255, 192, 203),  # Pink
        'book': (165, 42, 42),          # Brown
        'bottle': (0, 128, 0),          # Dark Green
        'cup': (128, 128, 0),           # Olive
        'tv': (255, 20, 147),           # Deep Pink
        'mouse': (50, 205, 50),         # Lime Green
        'keyboard': (186, 85, 211),     # Medium Orchid
        'default': (128, 128, 128)      # Gray
    }

    # --- SQLite DB Setup ---
    conn = sqlite3.connect('people_analytics.db', check_same_thread=False)
    c = conn.cursor()

    # Clear existing data on startup
    c.execute('DROP TABLE IF EXISTS people_count')
    c.execute('DROP TABLE IF EXISTS object_tracking')

    c.execute('''CREATE TABLE people_count (
        timestamp TEXT,
        count INTEGER
    )''')
    c.execute('''CREATE TABLE object_tracking (
        id INTEGER,
        timestamp TEXT,
        x INTEGER,
        y INTEGER,
        confidence REAL,
        time_in_frame REAL,
        object_class TEXT,
        object_name TEXT
    )''')
    conn.commit()

    # Camera controls
    st.sidebar.header("🎮 Controls")
    
    if st.sidebar.button("🎥 Start Camera", key="start_camera_btn"):
        st.session_state.camera_running = True
    
    if st.sidebar.button("⏹️ Stop Camera", key="stop_camera_btn"):
        st.session_state.camera_running = False

    # Video processing
    if st.session_state.camera_running:
        cap = cv2.VideoCapture(0)
        stframe = st.empty()
        
        while st.session_state.camera_running:
            ret, frame = cap.read()
            if not ret:
                st.error("❌ Failed to grab frame from webcam.")
                break
            
            # YOLO inference
            results = model(frame, device=device)
            detected_objects = results[0].boxes if results[0].boxes is not None else []

            current_time = time.time()
            current_objects = []
            
            # Process detected objects
            for i, box in enumerate(detected_objects):
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                confidence = float(box.conf[0])
                class_id = int(box.cls[0])
                object_name = COCO_CLASSES.get(class_id, f'class_{class_id}')
                cx, cy = (x1 + x2)//2, (y1 + y2)//2
                
                # Object tracking
                object_id = None
                min_distance = float('inf')
                
                for oid, (last_pos, last_class) in st.session_state.object_tracker.items():
                    if last_class == object_name:
                        distance = np.sqrt((cx - last_pos[0])**2 + (cy - last_pos[1])**2)
                        if distance < 100 and distance < min_distance:
                            min_distance = distance
                            object_id = oid
                
                if object_id is None:
                    object_id = st.session_state.next_object_id
                    st.session_state.next_object_id += 1
                    st.session_state.time_in_frame[object_id] = current_time
                
                st.session_state.object_tracker[object_id] = ((cx, cy), object_name)
                current_objects.append({
                    'id': object_id,
                    'bbox': (x1, y1, x2, y2),
                    'center': (cx, cy),
                    'confidence': confidence,
                    'time_in_frame': current_time - st.session_state.time_in_frame.get(object_id, current_time),
                    'class_id': class_id,
                    'object_name': object_name
                })

            # Draw bounding boxes and track paths
            for obj in current_objects:
                x1, y1, x2, y2 = obj['bbox']
                cx, cy = obj['center']
                object_id = obj['id']
                object_name = obj['object_name']
                
                color = CLASS_COLORS.get(object_name, CLASS_COLORS['default'])
                
                # Draw bounding box with object ID and name
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                cv2.putText(frame, f"{object_name} ID:{object_id}", (x1, y1-10), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
                cv2.putText(frame, f"Conf: {obj['confidence']:.2f}", (x1, y1-30), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 1)
                
                # Track path
                st.session_state.paths.setdefault(object_id, []).append((cx, cy))
                if len(st.session_state.paths[object_id]) > 50:
                    st.session_state.paths[object_id] = st.session_state.paths[object_id][-50:]
                
                # Draw path
                if len(st.session_state.paths[object_id]) > 1:
                    for j in range(1, len(st.session_state.paths[object_id])):
                        cv2.line(frame, st.session_state.paths[object_id][j-1], st.session_state.paths[object_id][j], color, 2)

                # Blur faces for people
                if object_name == 'person':
                    roi = frame[y1:y2, x1:x2]
                    if roi.size > 0:
                        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
                        faces = face_cascade.detectMultiScale(gray, 1.1, 4)
                        for (fx, fy, fw, fh) in faces:
                            face_roi = roi[fy:fy+fh, fx:fx+fw]
                            face_roi = cv2.GaussianBlur(face_roi, (99,99), 30)
                            roi[fy:fy+fh, fx:fx+fw] = face_roi
                
                # Log to database
                c.execute('INSERT INTO object_tracking (id, timestamp, x, y, confidence, time_in_frame, object_class, object_name) VALUES (?, ?, ?, ?, ?, ?, ?, ?)', 
                         (object_id, datetime.datetime.now().isoformat(), cx, cy, obj['confidence'], obj['time_in_frame'], obj['class_id'], object_name))

            # Clean up old trackers
            active_ids = [obj['id'] for obj in current_objects]
            st.session_state.object_tracker = {oid: data for oid, data in st.session_state.object_tracker.items() if oid in active_ids}

            total_objects = len(current_objects)
            people_count = len([obj for obj in current_objects if obj['object_name'] == 'person'])
            
            # Store tracking data
            st.session_state.tracking_history.append({
                'timestamp': datetime.datetime.now(),
                'count': people_count,
                'total_objects': total_objects,
                'objects': current_objects.copy()
            })

            # Log to DB
            now = datetime.datetime.now().isoformat()
            c.execute('INSERT INTO people_count (timestamp, count) VALUES (?, ?)', (now, total_objects))
            conn.commit()

            st.session_state.frame_count += 1

            # Display frame
            stframe.image(frame, channels="BGR", caption="🎥 Live Multi-Object Detection & Tracking")
            
            # Control frame rate
            time.sleep(0.03)
            
        cap.release()
    else:
        st.info("📷 Click 'Start' to begin live video feed")

elif page == "📊 Dashboard":
    # --- Dashboard Header ---
    st.markdown("""
        <div style='text-align: center; color: #003366; font-size: 2.2rem; font-weight: bold; letter-spacing: 1px; margin-bottom: 2rem;'>
            📊 Real-time Analytics Dashboard
        </div>
    """, unsafe_allow_html=True)

    # Status indicator
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        if st.session_state.get('camera_running', False):
            st.success("🟢 Live video is running")
        else:
            st.warning("🟡 Video feed is stopped")

    # Connect to database
    conn = sqlite3.connect('people_analytics.db', check_same_thread=False)
    
    # Debug information
    with st.expander("🔧 Debug Information"):
        st.write(f"Camera running: {st.session_state.get('camera_running', 'Not set')}")
        st.write(f"Tracking history length: {len(st.session_state.get('tracking_history', []))}")
        st.write(f"Object tracker size: {len(st.session_state.get('object_tracker', {}))}")
        st.write(f"Frame count: {st.session_state.get('frame_count', 0)}")
    
    # Get latest tracking data from session state
    if 'tracking_history' in st.session_state and st.session_state.tracking_history:
        latest_data = st.session_state.tracking_history
        
        # Real-time metrics row
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            current_total = latest_data[-1]['total_objects'] if latest_data else 0
            prev_total = latest_data[-2]['total_objects'] if len(latest_data) > 1 else 0
            st.metric("🎯 Current Objects", current_total, delta=current_total - prev_total)
        
        with col2:
            current_people = latest_data[-1]['count'] if latest_data else 0
            st.metric("👥 People Detected", current_people)
        
        with col3:
            if latest_data and latest_data[-1]['objects']:
                avg_conf = np.mean([obj['confidence'] for obj in latest_data[-1]['objects']])
                st.metric("🎯 Avg Confidence", f"{avg_conf:.1%}")
            else:
                st.metric("🎯 Avg Confidence", "0%")
        
        with col4:
            fps = len(latest_data) / max(1, (time.time() - (latest_data[0]['timestamp'].timestamp() if latest_data else time.time())))
            st.metric("📺 Processing Rate", f"{fps:.1f} FPS")

        # Charts row
        col1, col2 = st.columns(2)
        
        with col1:
            # Object count over time
            if len(latest_data) > 1:
                # Convert deque to list for slicing
                data_list = list(latest_data)
                times = [d['timestamp'] for d in data_list[-50:]]
                counts = [d['total_objects'] for d in data_list[-50:]]
                
                fig_count = go.Figure()
                fig_count.add_trace(go.Scatter(
                    x=times, y=counts,
                    mode='lines+markers',
                    name='Object Count',
                    line=dict(color='#00AEEF', width=3),
                    marker=dict(size=6)
                ))
                fig_count.update_layout(
                    title="📊 Real-time Object Count",
                    xaxis_title="Time",
                    yaxis_title="Count",
                    height=400,
                    showlegend=False
                )
                st.plotly_chart(fig_count, use_container_width=True)
        
        with col2:
            # Current object types distribution
            if latest_data and latest_data[-1]['objects']:
                current_objects = latest_data[-1]['objects']
                object_types = {}
                for obj in current_objects:
                    obj_type = obj['object_name']
                    object_types[obj_type] = object_types.get(obj_type, 0) + 1
                
                if object_types:
                    fig_pie = go.Figure(data=[go.Pie(
                        labels=list(object_types.keys()),
                        values=list(object_types.values()),
                        hole=0.3
                    )])
                    fig_pie.update_layout(
                        title="🎯 Current Object Distribution",
                        height=400
                    )
                    st.plotly_chart(fig_pie, use_container_width=True)
            else:
                st.info("📊 No objects currently detected")

        # Movement tracking
        st.markdown("### 🗺️ Movement Tracking")
        col1, col2 = st.columns(2)
        
        with col1:
            # Object tracking table
            if latest_data and latest_data[-1]['objects']:
                df_objects = pd.DataFrame([
                    {
                        'ID': obj['id'],
                        'Object': obj['object_name'],
                        'Confidence': f"{obj['confidence']:.1%}",
                        'Time (s)': f"{obj['time_in_frame']:.1f}",
                        'Position': f"({obj['center'][0]}, {obj['center'][1]})"
                    }
                    for obj in latest_data[-1]['objects']
                ])
                st.dataframe(df_objects, use_container_width=True, height=300)
            else:
                st.info("📋 No active objects to track")
        
        with col2:
            # Activity heatmap
            if len(latest_data) > 5:
                recent_positions = []
                # Convert deque to list for slicing
                data_list = list(latest_data)
                for frame_data in data_list[-10:]:
                    for obj in frame_data['objects']:
                        recent_positions.append(obj['center'])
                
                if recent_positions:
                    # Create simple heatmap visualization
                    x_coords = [pos[0] for pos in recent_positions]
                    y_coords = [pos[1] for pos in recent_positions]
                    
                    fig_heat = go.Figure(data=go.Histogram2d(
                        x=x_coords,
                        y=y_coords,
                        colorscale='Hot',
                        showscale=False
                    ))
                    fig_heat.update_layout(
                        title="🔥 Activity Heatmap",
                        xaxis_title="X Position",
                        yaxis_title="Y Position",
                        height=300
                    )
                    st.plotly_chart(fig_heat, use_container_width=True)
                else:
                    st.info("🗺️ No movement data available")
            else:
                st.info("🗺️ Collecting movement data...")

        # Auto-refresh dashboard
        st.markdown("---")
        col1, col2, col3 = st.columns([1, 1, 1])
        with col2:
            if st.button("🔄 Refresh Dashboard", use_container_width=True):
                st.rerun()
        
        # Auto refresh every 2 seconds when camera is running
        if st.session_state.get('camera_running', False):
            time.sleep(2)
            st.rerun()
    
    else:
        # No live session data, check database for historical data
        c = conn.cursor()
        c.execute('SELECT COUNT(*) FROM people_count')
        db_records = c.fetchone()[0]
        
        if db_records > 0:
            st.info(f"📊 Showing historical data from database ({db_records} records)")
            
            # Historical metrics
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                c.execute('SELECT COUNT(*) FROM people_count')
                total_frames = c.fetchone()[0]
                st.metric("📊 Total Frames", total_frames)
            
            with col2:
                c.execute('SELECT COUNT(DISTINCT id) FROM object_tracking')
                unique_objects = c.fetchone()[0]
                st.metric("🎯 Unique Objects", unique_objects)
            
            with col3:
                c.execute('SELECT AVG(confidence) FROM object_tracking WHERE confidence > 0')
                avg_conf = c.fetchone()[0]
                if avg_conf:
                    st.metric("🎯 Avg Confidence", f"{avg_conf:.1%}")
                else:
                    st.metric("🎯 Avg Confidence", "N/A")
            
            with col4:
                c.execute('SELECT MAX(count) FROM people_count')
                max_objects = c.fetchone()[0]
                st.metric("📈 Peak Objects", max_objects or 0)
            
            # Historical charts
            col1, col2 = st.columns(2)
            
            with col1:
                # Object count history from database
                df_history = pd.read_sql_query('SELECT * FROM people_count ORDER BY timestamp DESC LIMIT 100', conn)
                if not df_history.empty:
                    df_history['timestamp'] = pd.to_datetime(df_history['timestamp'])
                    fig_history = go.Figure()
                    fig_history.add_trace(go.Scatter(
                        x=df_history['timestamp'],
                        y=df_history['count'],
                        mode='lines+markers',
                        name='Object Count',
                        line=dict(color='#00AEEF', width=2),
                        marker=dict(size=4)
                    ))
                    fig_history.update_layout(
                        title="📊 Historical Object Count (Last 100 frames)",
                        xaxis_title="Time",
                        yaxis_title="Object Count",
                        height=400,
                        showlegend=False
                    )
                    st.plotly_chart(fig_history, use_container_width=True)
            
            with col2:
                # Object type distribution from database
                df_objects = pd.read_sql_query('SELECT object_name, COUNT(*) as count FROM object_tracking GROUP BY object_name ORDER BY count DESC', conn)
                if not df_objects.empty:
                    fig_dist = go.Figure(data=[go.Pie(
                        labels=df_objects['object_name'],
                        values=df_objects['count'],
                        hole=0.3
                    )])
                    fig_dist.update_layout(
                        title="🎯 Historical Object Distribution",
                        height=400
                    )
                    st.plotly_chart(fig_dist, use_container_width=True)
            
            # Recent activity table
            st.markdown("### 📋 Recent Activity")
            df_recent = pd.read_sql_query('''
                SELECT object_name, confidence, timestamp, 
                       CASE WHEN object_name = "person" THEN "👥" 
                            WHEN object_name = "car" THEN "🚗"
                            WHEN object_name = "bicycle" THEN "🚲"
                            ELSE "📦" END as icon
                FROM object_tracking 
                ORDER BY timestamp DESC 
                LIMIT 20
            ''', conn)
            
            if not df_recent.empty:
                df_recent['confidence'] = df_recent['confidence'].apply(lambda x: f"{x:.1%}")
                df_recent['timestamp'] = pd.to_datetime(df_recent['timestamp']).dt.strftime('%H:%M:%S')
                st.dataframe(df_recent, use_container_width=True, height=300)
        
        else:
            st.info("📊 No tracking data available. Start the video feed to see dashboard analytics.")
        
        # Navigation buttons
        col1, col2 = st.columns(2)
        with col1:
            if st.button("🎥 Go to Live Video", use_container_width=True):
                st.switch_page("🎥 Live Video")
        with col2:
            if st.button("🔄 Check for Data", use_container_width=True):
                st.rerun()

else:  # Reports & Analytics page
    st.markdown("## 📊 Analytics & Reports Dashboard")
    
    # Connect to database
    conn = sqlite3.connect('people_analytics.db', check_same_thread=False)
    
    # Display database stats
    col1, col2, col3 = st.columns(3)
    
    with col1:
        # Get total records in people_count table
        c = conn.cursor()
        c.execute('SELECT COUNT(*) FROM people_count')
        total_records = c.fetchone()[0]
        st.metric("📊 Total Frame Records", total_records)
    
    with col2:
        # Get total individual objects tracked
        c.execute('SELECT COUNT(DISTINCT id) FROM object_tracking')
        unique_objects = c.fetchone()[0]
        st.metric("🎯 Unique Objects Tracked", unique_objects)
    
    with col3:
        # Get average confidence
        c.execute('SELECT AVG(confidence) FROM object_tracking WHERE confidence > 0')
        avg_conf = c.fetchone()[0]
        if avg_conf:
            st.metric("🎯 Average Confidence", f"{avg_conf:.1%}")
        else:
            st.metric("🎯 Average Confidence", "N/A")

    # Historical data charts
    st.markdown("### 📈 Historical Trends")
    
    # Get historical data
    df_history = pd.read_sql_query('SELECT * FROM people_count ORDER BY timestamp', conn)
    df_objects = pd.read_sql_query('SELECT * FROM object_tracking ORDER BY timestamp', conn)
    
    if not df_history.empty:
        col1, col2 = st.columns(2)
        
        with col1:
            # Object count over time
            df_history['timestamp'] = pd.to_datetime(df_history['timestamp'])
            fig_history = go.Figure()
            fig_history.add_trace(go.Scatter(
                x=df_history['timestamp'],
                y=df_history['count'],
                mode='lines',
                name='Object Count',
                line=dict(color='#00AEEF', width=2)
            ))
            fig_history.update_layout(
                title="📊 Object Detection History",
                xaxis_title="Time",
                yaxis_title="Object Count",
                height=400
            )
            st.plotly_chart(fig_history, use_container_width=True)
        
        with col2:
            # Object type distribution
            if not df_objects.empty:
                object_counts = df_objects['object_name'].value_counts()
                fig_dist = go.Figure(data=[go.Bar(
                    x=object_counts.index,
                    y=object_counts.values,
                    marker_color='#00AEEF'
                )])
                fig_dist.update_layout(
                    title="🎯 Object Types Detected",
                    xaxis_title="Object Type",
                    yaxis_title="Detection Count",
                    height=400
                )
                st.plotly_chart(fig_dist, use_container_width=True)
    
    # Raw data tables
    st.markdown("### 📋 Raw Data")
    
    tab1, tab2 = st.tabs(["Frame Summary", "Object Details"])
    
    with tab1:
        if not df_history.empty:
            st.dataframe(df_history.tail(100), use_container_width=True)
        else:
            st.info("No frame data available yet.")
    
    with tab2:
        if not df_objects.empty:
            st.dataframe(df_objects.tail(100), use_container_width=True)
        else:
            st.info("No object tracking data available yet.")
    
    conn.close()
