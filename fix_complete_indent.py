#!/usr/bin/env python3
"""
Fix streamlit_app.py indentation completely
"""

with open('streamlit_app.py.backup', 'r') as f:
    lines = f.readlines()

# Find key markers
sidebar_end = None
data_processing_if = None
database_explorer_elif = None

for i, line in enumerate(lines):
    if 'st.sidebar.markdown("---")' in line and sidebar_end is None:
        # Find the LAST occurrence
        sidebar_end = i
    if 'if st.session_state.app_mode == "📁 Data Processing"' in line:
        data_processing_if = i
    if 'elif st.session_state.app_mode == "🔍 Database Explorer"' in line:
        database_explorer_elif = i

if sidebar_end is None or data_processing_if is None or database_explorer_elif is None:
    print("ERROR: Could not find markers")
    print(f"sidebar_end: {sidebar_end}, data_processing_if: {data_processing_if}, database_explorer_elif: {database_explorer_elif}")
    exit(1)

print(f"Sidebar end at line {sidebar_end + 1}")
print(f"Data Processing if at line {data_processing_if + 1}")
print(f"Database Explorer elif at line {database_explorer_elif + 1}")

# Write output
output = []

# Part 1: Everything up to and including sidebar end (keep as is)
for i in range(sidebar_end + 1):
    output.append(lines[i])

# Part 2: Between sidebar and Data Processing (keep as is - comments)
for i in range(sidebar_end + 1, data_processing_if):
    output.append(lines[i])

# Part 3: Data Processing section (indent content by 4 spaces)
output.append(lines[data_processing_if])  # Keep "if" line as is
for i in range(data_processing_if + 1, database_explorer_elif):
    line = lines[i]
    if line.strip():  # Non-empty
        output.append('    ' + line)
    else:  # Empty
        output.append(line)

# Part 4: Database Explorer section (indent content by 4 spaces)  
output.append(lines[database_explorer_elif])  # Keep "elif" line as is
for i in range(database_explorer_elif + 1, len(lines)):
    line = lines[i]
    if line.strip():  # Non-empty
        output.append('    ' + line)
    else:  # Empty
        output.append(line)

# Write
with open('streamlit_app.py', 'w') as f:
    f.writelines(output)

print(f"✓ Wrote {len(output)} lines")
