#!/usr/bin/env python3
#
# Permission is  hereby  granted,  free  of  charge,  to  any  person
# obtaining a copy of  this  software  and  associated  documentation
# files  (the  "Software"),  to  deal   in   the   Software   without
# restriction, including without limitation the rights to use,  copy,
# modify, merge, publish, distribute, sublicense, and/or sell  copies
# of the Software, and to permit persons  to  whom  the  Software  is
# furnished to do so.
# 
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT  WARRANTY  OF  ANY  KIND,
# EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES  OF
# MERCHANTABILITY,   FITNESS   FOR   A   PARTICULAR    PURPOSE    AND
# NONINFRINGEMENT.  IN  NO  EVENT  SHALL  THE  AUTHORS  OR  COPYRIGHT
# OWNER(S) BE LIABLE FOR  ANY  CLAIM,  DAMAGES  OR  OTHER  LIABILITY,
# WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING  FROM,
# OUT OF OR IN CONNECTION WITH THE  SOFTWARE  OR  THE  USE  OR  OTHER
# DEALINGS IN THE SOFTWARE.
#
##########################License above is Equivalent to Public Domain

import os
import sys

# example ./concat.py ./ file:file1.txt file:COA.h type:c
# ./concat.py ./ file:COA.h file:COAtest.c

def escape_markdown_code_blocks(lines):
    """Escape triple backticks in content to prevent markdown conflicts."""
    escaped_lines = []
    for line in lines:
        # Replace ``` with \``` so it doesn't break our code blocks
        escaped_line = line.replace("```", "\\```")
        escaped_lines.append(escaped_line)
    return escaped_lines

def list_files_and_types(directory):
    """List all files and their types in the directory recursively."""
    file_types = {}
    file_names = []
    
    print(f"\nListing files in directory: {directory}\n")
    
    # Walk through directory recursively
    for root, dirs, files in os.walk(directory):
        for filename in files:
            # Skip files that might be this script itself
            if filename == "concat.py":
                continue
                
            name, ext = os.path.splitext(filename)
            ext = ext[1:] if ext else ""  # Remove dot from extension
            
            # Add to file names list
            file_path = os.path.join(root, filename)
            file_names.append(file_path)
            
            # Track file types
            if ext:
                if ext not in file_types:
                    file_types[ext] = []
                file_types[ext].append(file_path)
    
    # Print all files
    print("All files:")
    for file_path in sorted(file_names):
        print(f"  {file_path}")
    
    # Print files by type
    print("\nFiles by type:")
    for ext, files in sorted(file_types.items()):
        print(f"\n  .{ext} files:")
        for file_path in sorted(files):
            print(f"    {file_path}")
    
    return file_names, file_types

def main():
    if len(sys.argv) < 2:
        print("Usage: python concat.py <directory> [--list] [file:filename1] [file:filename2] ... [type:FileType1] [type:FileType2] ...")
        print("\nOptions:")
        print("  --list                : List all files and their types in the directory")
        print("  file:filename         : Specify a particular file to include")
        print("  type:FileType         : Specify a file type/extension to include all files of that type")
        return

    directory = sys.argv[1]
    HR_len = 5
    
    # Check if directory exists
    if not os.path.isdir(directory):
        print(f"Error: Directory '{directory}' not found")
        return
    
    # List files if --list is provided
    if "--list" in sys.argv:
        list_files_and_types(directory)
        return
    
    # Parse remaining arguments to identify files and types
    file_targets = []
    type_targets = []
    
    for arg in sys.argv[2:]:
        if arg.startswith("file:"):
            # This is a specific file
            file_path = arg[5:]  # Remove "file:" prefix
            # Make it a relative path from the directory
            if not os.path.isabs(file_path):
                file_path = os.path.join(directory, file_path)
            file_targets.append(file_path)
        elif arg.startswith("type:"):
            # This is a file type/extension
            file_type = arg[5:]  # Remove "type:" prefix
            type_targets.append(file_type)
        else:
            # For backward compatibility, treat as file type
            type_targets.append(arg)
    
    # If no targets specified, show help
    if not file_targets and not type_targets:
        print("Error: No files or file types specified")
        print("Use --list to see available files, or specify files with 'file:filename' or types with 'type:extension'")
        return
    
    output_lines = []
    output_lines.append(f"{'#' * HR_len}\n")
    output_lines.append(f"occurences of ``` will be escaped for markdown\n\n")
    
    # Process specific files
    for file_path in file_targets:
        if not os.path.isfile(file_path):
            print(f"Warning: File '{file_path}' not found, skipping")
            continue
            
        # Get just the filename for the header
        filename = os.path.basename(file_path)
        
        # Read contents of file
        try:
            with open(file_path, 'r') as infile:
                lines = infile.readlines()
                if lines:
                    # Add filename as header
                    output_lines.append(f"\n{'#' * HR_len}\n")
                    output_lines.append(f"File: {filename}\n")
                    output_lines.append(f"{'#' * HR_len}\n")
                    output_lines.append(f"\n{'`' * 3}\n")
                    
                    # Escape any triple backticks in the content
                    escaped_lines = escape_markdown_code_blocks(lines)
                    output_lines.extend(escaped_lines)
                    
                    output_lines.append(f"\n{'`' * 3}\n")
        except Exception as e:
            # Skip files that can't be opened
            print(f"Could not read {filename}: {str(e)}")
    
    # Process files by type
    for root, dirs, files in os.walk(directory):
        for filename in files:
            # Skip files that might be this script itself
            if filename == "concat.py":
                continue
                
            name, ext = os.path.splitext(filename)
            ext = ext[1:]  # Remove dot from extension
            
            # Check if extension matches any of our file types
            if ext in type_targets:
                # Build full path to file
                file_path = os.path.join(root, filename)
                
                # Read contents of file
                try:
                    with open(file_path, 'r') as infile:
                        lines = infile.readlines()
                        if lines:
                            # Add filename as header
                            output_lines.append(f"\n{'#' * HR_len}\n")
                            output_lines.append(f"File: {filename}\n")
                            output_lines.append(f"{'#' * HR_len}\n")
                            output_lines.append(f"\n{'`' * 3}\n")
                            
                            # Escape any triple backticks in the content
                            escaped_lines = escape_markdown_code_blocks(lines)
                            output_lines.extend(escaped_lines)
                            
                            output_lines.append(f"\n{'`' * 3}\n")
                except Exception as e:
                    # Skip files that can't be opened
                    print(f"Could not read {filename}: {str(e)}")

    # Write to output file
    output_path = os.path.join(directory, "concat.md")
    with open(output_path, 'w') as outfile:
        outfile.writelines(output_lines)
    
    print(f"Successfully concatenated files to {output_path}")
    print(f"Directory: {directory}")

if __name__ == "__main__":
    main()