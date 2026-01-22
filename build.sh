#!/bin/bash
set -e

# Pandoc version (can be overridden via environment variable)
PANDOC_VERSION="${PANDOC_VERSION:-3.1.1}"

# Create output directory
mkdir -p _site

# Install pandoc if not available
if ! command -v pandoc &> /dev/null; then
    echo "Installing pandoc ${PANDOC_VERSION}..."
    curl -L "https://github.com/jgm/pandoc/releases/download/${PANDOC_VERSION}/pandoc-${PANDOC_VERSION}-linux-amd64.tar.gz" -o pandoc.tar.gz
    tar xvzf pandoc.tar.gz
    export PATH="$PWD/pandoc-${PANDOC_VERSION}/bin:$PATH"
fi

echo "Pandoc version: $(pandoc --version | head -1)"

# Copy CSS to output
cp assets/style.css _site/
cp assets/diff-style.css _site/

# Copy images to output
mkdir -p _site/comparisons
cp comparisons/*.png _site/comparisons/

# Convert LaTeX to HTML using pandoc
echo "Converting manuscript to HTML..."
pandoc manuscript/main.tex \
    --from=latex \
    --to=html5 \
    --standalone \
    --mathjax \
    --number-sections \
    --lua-filter=assets/fix-refs.lua \
    --css=style.css \
    --metadata title="A new tail bound for the sum of bounded independent random variables" \
    --citeproc \
    --bibliography=manuscript/bibliography.bib \
    --csl=assets/apa.csl \
    --output=_site/index.html

echo "Build complete! Output in _site/"

# ============================================================
# Generate diff HTML comparing current branch to main branch
# ============================================================
echo ""
echo "=== Generating diff against main branch ==="

DIFF_GENERATED=false

# Check if we're in a git repository and can access main branch
if git rev-parse --git-dir > /dev/null 2>&1; then
    echo "Git repository detected. Current branch: ${BRANCH:-$(git rev-parse --abbrev-ref HEAD)}"
    
    # Fetch main branch for comparison (Netlify does a shallow clone)
    echo "Fetching main branch for comparison..."
    git fetch origin main:refs/remotes/origin/main --depth=1 2>/dev/null || {
        echo "Warning: Could not fetch main branch. Trying alternative fetch..."
        git fetch --unshallow 2>/dev/null || git fetch origin main --depth=1 2>/dev/null || true
    }
    
    # Try to get the main branch version of manuscript
    if git show origin/main:manuscript/main.tex > /tmp/main_manuscript.tex 2>/dev/null; then
        echo "Successfully retrieved main branch manuscript."
        
        # Also get the bibliography from main branch
        git show origin/main:manuscript/bibliography.bib > /tmp/main_bibliography.bib 2>/dev/null || cp manuscript/bibliography.bib /tmp/main_bibliography.bib
        
        # Build HTML from main branch manuscript
        echo "Building HTML from main branch..."
        pandoc /tmp/main_manuscript.tex \
            --from=latex \
            --to=html5 \
            --standalone \
            --mathjax \
            --number-sections \
            --lua-filter=assets/fix-refs.lua \
            --css=style.css \
            --metadata title="A new tail bound for the sum of bounded independent random variables" \
            --citeproc \
            --bibliography=/tmp/main_bibliography.bib \
            --csl=assets/apa.csl \
            --output=/tmp/main_index.html 2>/dev/null && {
                
            echo "Generating HTML diff..."
            
            # Create a unified diff of the HTML content (body only, stripped of tags for readability)
            # Extract just the body content for comparison
            sed -n '/<body/,/<\/body>/p' /tmp/main_index.html | sed 's/<[^>]*>//g' | sed 's/^[[:space:]]*//;s/[[:space:]]*$//' | grep -v '^$' > /tmp/main_text.txt || true
            sed -n '/<body/,/<\/body>/p' _site/index.html | sed 's/<[^>]*>//g' | sed 's/^[[:space:]]*//;s/[[:space:]]*$//' | grep -v '^$' > /tmp/current_text.txt || true
            
            # Generate diff HTML page
            {
                echo '<!DOCTYPE html>'
                echo '<html lang="en">'
                echo '<head>'
                echo '  <meta charset="UTF-8">'
                echo '  <meta name="viewport" content="width=device-width, initial-scale=1.0">'
                echo '  <title>Manuscript Changes (Diff vs Main)</title>'
                echo '  <link rel="stylesheet" href="diff-style.css">'
                echo '</head>'
                echo '<body>'
                echo '  <h1>Manuscript Changes (Diff vs Main Branch)</h1>'
                echo '  <p class="diff-info">This page shows the differences between the current branch and the main branch.</p>'
                echo '  <p style="text-align: center;"><a href="/">← Back to manuscript</a></p>'
                echo '  <div class="diff-legend">'
                echo '    <span class="diff-added">+ Added lines</span>'
                echo '    <span class="diff-removed">- Removed lines</span>'
                echo '  </div>'
                echo '  <pre class="diff-content">'
                # Generate unified diff and escape HTML entities
                diff -u /tmp/main_text.txt /tmp/current_text.txt 2>/dev/null | \
                    sed 's/&/\&amp;/g; s/</\&lt;/g; s/>/\&gt;/g' | \
                    sed 's/^-\(.*\)$/<span class="line-removed">-\1<\/span>/' | \
                    sed 's/^+\(.*\)$/<span class="line-added">+\1<\/span>/' | \
                    sed 's/^@\(.*\)$/<span class="line-info">@\1<\/span>/' || echo "No differences found or diff failed"
                echo '  </pre>'
                echo '</body>'
                echo '</html>'
            } > _site/diff.html
            DIFF_GENERATED=true
            echo "Diff page generated successfully."
        } || {
            echo "Warning: Could not build HTML from main branch manuscript"
        }
    else
        echo "Warning: Could not retrieve main branch manuscript"
    fi
else
    echo "Not a git repository or git not available"
fi

# Create a placeholder diff page if no diff was generated
if [ "$DIFF_GENERATED" = false ]; then
    echo "Creating placeholder diff page..."
    {
        echo '<!DOCTYPE html>'
        echo '<html lang="en">'
        echo '<head>'
        echo '  <meta charset="UTF-8">'
        echo '  <meta name="viewport" content="width=device-width, initial-scale=1.0">'
        echo '  <title>Manuscript Diff - Not Available</title>'
        echo '  <link rel="stylesheet" href="diff-style.css">'
        echo '</head>'
        echo '<body>'
        echo '  <h1>Manuscript Diff</h1>'
        echo '  <p class="diff-info">Diff is only available for pull request previews or branch deploys.</p>'
        echo '  <p>On the main branch, there is no diff to show.</p>'
        echo '  <p><a href="/">← Back to manuscript</a></p>'
        echo '</body>'
        echo '</html>'
    } > _site/diff.html
fi

echo ""
echo "=== Build complete! ==="
echo "Main manuscript: _site/index.html"
echo "Diff page: _site/diff.html"
