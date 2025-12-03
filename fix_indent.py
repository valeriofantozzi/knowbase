#!/usr/bin/env python3
"""
Script to fix indentation in streamlit_app.py
"""

# Read the original backup
with open('streamlit_app.py.backup', 'r') as f:
    lines = f.readlines()

# Find where to split (after st.sidebar.markdown("---"))
split_idx = None
for i in range(len(lines) - 1, -1, -1):
    if 'st.sidebar.markdown("---")' in lines[i]:
        split_idx = i + 1
        break

if not split_idx:
    print("Could not find split point")
    exit(1)

# Write new file
with open('streamlit_app.py', 'w') as f:
    # Write everything up to split point
    for i in range(split_idx):
        f.write(lines[i])
    
    # Add new mode structure
    f.write("""
# ============================================================================
# MAIN CONTENT
# ============================================================================

# ============================================================================
# MODE 1: DATA PROCESSING
# ============================================================================
if st.session_state.app_mode == "📁 Data Processing":
    st.title("📁 Data Processing & Vectorization")
    st.markdown("Upload subtitle files and generate embeddings for vector database indexing")
    
    # File Upload Section
    st.header("📤 Upload Subtitle Files")
    
    col_upload1, col_upload2 = st.columns([2, 1])
    
    with col_upload1:
        uploaded_files = st.file_uploader(
            "Upload SRT files",
            type=['srt'],
            accept_multiple_files=True,
            help="Select one or more .srt subtitle files to process"
        )
        
        if uploaded_files:
            st.success(f"✓ {len(uploaded_files)} file(s) uploaded")
            
            # Save uploaded files temporarily
            upload_dir = Path("data/raw/subtitles")
            upload_dir.mkdir(parents=True, exist_ok=True)
            
            for uploaded_file in uploaded_files:
                file_path = upload_dir / uploaded_file.name
                with open(file_path, "wb") as fp:
                    fp.write(uploaded_file.getbuffer())
            
            st.info(f"Files saved to: {upload_dir}")
    
    with col_upload2:
        # Scan existing files
        scan_dir = Path("data/raw/subtitles")
        if scan_dir.exists():
            existing_files = list(scan_dir.rglob("*.srt"))
            st.metric("Files in directory", len(existing_files))
            
            if st.button("🔍 Scan Directory", use_container_width=True):
                st.rerun()
        else:
            st.metric("Files in directory", 0)
    
    # File Selection Section
    st.header("📋 Available Files")
    
    scan_dir = Path("data/raw/subtitles")
    if scan_dir.exists():
        available_files = list(scan_dir.rglob("*.srt"))
        
        if available_files:
            st.markdown(f"**Found {len(available_files)} SRT files**")
            
            # File table
            file_data = []
            for file_path in available_files:
                file_stat = file_path.stat()
                file_data.append({
                    'Filename': file_path.name,
                    'Size (KB)': f"{file_stat.st_size / 1024:.2f}",
                })
            
            df = pd.DataFrame(file_data)
            st.dataframe(df, use_container_width=True, height=300)
            
            st.info("📝 **Note:** Processing interface will be implemented in the next update")
        else:
            st.warning("No SRT files found. Upload files above.")
    else:
        st.warning(f"Directory {scan_dir} does not exist. Upload files to create it.")

# ============================================================================
# MODE 2: DATABASE EXPLORER
# ============================================================================
elif st.session_state.app_mode == "🔍 Database Explorer":
""")
    
    # Now write the rest with proper indentation
    # Find where "# Main content" starts in the original
    main_content_idx = None
    for i in range(split_idx, len(lines)):
        if '# Main content' in lines[i]:
            main_content_idx = i
            break
    
    if main_content_idx:
        # Skip the original "# Main content" line and following title lines
        idx = main_content_idx
        while idx < len(lines) and ('# Main content' in lines[idx] or 
                                     'st.title("📊 Vector Database Visualization")' in lines[idx] or
                                     'st.markdown("Explore your subtitle embeddings database interactively")' in lines[idx]):
            idx += 1
        
        # Add title lines with indentation
        f.write('    st.title("📊 Vector Database Visualization")\n')
        f.write('    st.markdown("Explore your subtitle embeddings database interactively")\n')
        f.write('\n')
        
        # Write rest with indentation
        for i in range(idx, len(lines)):
            line = lines[i]
            if line.strip():  # Non-empty line
                f.write('    ' + line)
            else:  # Empty line
                f.write('\n')

print("File fixed successfully!")
