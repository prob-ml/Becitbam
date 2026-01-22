#!/bin/bash
set -e

# Create output directory
mkdir -p _site

# Install pandoc if not available
if ! command -v pandoc &> /dev/null; then
    echo "Installing pandoc..."
    curl -L https://github.com/jgm/pandoc/releases/download/3.1.1/pandoc-3.1.1-linux-amd64.tar.gz -o pandoc.tar.gz
    tar xvzf pandoc.tar.gz
    export PATH="$PWD/pandoc-3.1.1/bin:$PATH"
fi

echo "Pandoc version: $(pandoc --version | head -1)"

# Copy CSS to output
cp assets/style.css _site/

# Convert LaTeX to HTML using pandoc
echo "Converting manuscript to HTML..."
pandoc manuscript/main.tex \
    --from=latex \
    --to=html5 \
    --standalone \
    --mathjax \
    --css=style.css \
    --metadata title="A new tail bound for the sum of bounded independent random variables" \
    --citeproc \
    --bibliography=manuscript/bibliography.bib \
    --csl=assets/apa.csl \
    --output=_site/index.html

echo "Build complete! Output in _site/"
