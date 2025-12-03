#!/usr/bin/env python3
"""
Fix indentation in streamlit_app.py for Database Explorer section
"""

# Read the file
with open('streamlit_app.py', 'r') as f:
    lines = f.readlines()

# Find the elif line
elif_idx = None
for i, line in enumerate(lines):
    if 'elif st.session_state.app_mode == "🔍 Database Explorer"' in line:
        elif_idx = i
        break

if elif_idx is None:
    print("ERROR: Could not find elif statement")
    exit(1)

print(f"Found elif at line {elif_idx + 1}")

# Keep everything up to and including elif
output = lines[:elif_idx + 1]

# Process lines after elif
current_line = elif_idx + 1
while current_line < len(lines):
    line = lines[current_line]
    
    # Empty lines - keep as is
    if not line.strip():
        output.append(line)
        current_line += 1
        continue
    
    # Already indented with at least 4 spaces - keep as is
    if line.startswith('    '):
        output.append(line)
        current_line += 1
        continue
    
    # Not indented - add 4 spaces
    output.append('    ' + line)
    current_line += 1

# Write output
with open('streamlit_app.py', 'w') as f:
    f.writelines(output)

print(f"✓ Fixed {len(output)} total lines")
