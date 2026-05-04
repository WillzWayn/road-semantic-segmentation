#!/bin/bash

# Compile script for report and presentation
# Compiles both report.tex and presentation.tex, then moves PDFs to output/

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SRC_DIR="$SCRIPT_DIR/src"
OUTPUT_DIR="$SCRIPT_DIR/output"

# Colors for output
GREEN='\033[0;32m'
BLUE='\033[0;34m'
RED='\033[0;31m'
YELLOW='\033[0;33m'
NC='\033[0m' # No Color

# Function to compile a LaTeX file
compile_latex() {
    local tex_file=$1
    local pdf_name=$2

    echo -e "${BLUE}Compiling $tex_file...${NC}"

    cd "$SRC_DIR"

    # First pass
    if ! pdflatex -interaction=nonstopmode "$tex_file" > compile_temp.log 2>&1; then
        echo -e "${RED}Error: First pass failed for $tex_file${NC}"
        echo "Last 30 lines of log:"
        tail -30 compile_temp.log
        rm -f compile_temp.log
        return 1
    fi
    rm -f compile_temp.log

    # Run bibtex if .bib file exists
    if [ -f "report.bib" ] && [ "$tex_file" = "report.tex" ]; then
        bibtex report > /dev/null 2>&1 || true
    fi

    # Second pass for cross-references
    if ! pdflatex -interaction=nonstopmode "$tex_file" > compile_temp.log 2>&1; then
        echo -e "${RED}Error: Second pass failed for $tex_file${NC}"
        echo "Last 30 lines of log:"
        tail -30 compile_temp.log
        rm -f compile_temp.log
        return 1
    fi
    rm -f compile_temp.log

    # Third pass to ensure all references are resolved
    if ! pdflatex -interaction=nonstopmode "$tex_file" > compile_temp.log 2>&1; then
        echo -e "${RED}Error: Third pass failed for $tex_file${NC}"
        echo "Last 30 lines of log:"
        tail -30 compile_temp.log
        rm -f compile_temp.log
        return 1
    fi
    rm -f compile_temp.log

    # Move PDF to output directory
    if [ -f "${tex_file%.tex}.pdf" ]; then
        mv "${tex_file%.tex}.pdf" "$OUTPUT_DIR/$pdf_name"
        echo -e "${GREEN}✓ $pdf_name created${NC}"
    else
        echo -e "${RED}Error: PDF not generated for $tex_file${NC}"
        return 1
    fi

    cd "$SCRIPT_DIR"
    return 0
}

# Main execution
echo -e "${BLUE}=== LaTeX Compilation Script ===${NC}"

# Compile report
if ! compile_latex "report.tex" "report.pdf"; then
    echo -e "${RED}=== Compilation Failed ===${NC}"
    exit 1
fi

# Compile presentation
if ! compile_latex "presentation.tex" "presentation.pdf"; then
    echo -e "${RED}=== Compilation Failed ===${NC}"
    exit 1
fi

# Clean up auxiliary files in src/
echo -e "${BLUE}Cleaning up auxiliary files...${NC}"
cd "$SRC_DIR"
rm -f *.aux *.log *.out *.nav *.snm *.toc *.blg *.bbl *.synctex.gz
cd "$SCRIPT_DIR"

echo -e "${GREEN}=== Compilation Complete ===${NC}"
echo -e "PDFs available in: ${BLUE}$OUTPUT_DIR/${NC}"
ls -lh "$OUTPUT_DIR"/*.pdf
