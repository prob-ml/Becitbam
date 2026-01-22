-- Lua filter to fix LaTeX equation references for HTML output
-- 1. Adds id anchors to equations with \label
-- 2. Adds equation numbers (1), (2), etc. to labeled equations
-- 3. Converts unresolved references to proper numbered links

local equation_numbers = {}
local current_eq = 0

-- First pass: collect equation labels, assign numbers, and add anchors/numbers
function Math(el)
  if el.mathtype == "DisplayMath" then
    local label = el.text:match("\\label{([^}]+)}")
    if label then
      current_eq = current_eq + 1
      equation_numbers[label] = current_eq
      
      -- Remove the \label from the math content (MathJax doesn't need it)
      local clean_math = el.text:gsub("\\label{[^}]+}", "")
      
      -- Create the equation with anchor and number
      -- Wrap in a div with the anchor id, then include the math and the number
      return {
        pandoc.RawInline("html", '<span id="' .. label .. '" class="equation-wrapper" style="display:flex;justify-content:center;align-items:center;gap:1rem;">'),
        pandoc.Math("DisplayMath", clean_math),
        pandoc.RawInline("html", '<span class="equation-number">(' .. current_eq .. ')</span></span>')
      }
    end
  end
  return el
end

-- Second pass: fix references in links
function Link(el)
  local target = el.target
  if target:match("^#eq:") or target:match("^#thm:") or target:match("^#fig:") or target:match("^#sec:") then
    local label = target:sub(2)  -- Remove the leading #
    local num = equation_numbers[label]
    if num then
      el.content = {pandoc.Str("(" .. num .. ")")}
    end
  end
  return el
end

-- Also handle Span elements which may contain unresolved references
function Span(el)
  if el.attributes["data-reference-type"] == "ref" then
    local label = el.attributes["data-reference"]
    if label then
      local num = equation_numbers[label]
      if num then
        return pandoc.Link(
          {pandoc.Str("(" .. num .. ")")},
          "#" .. label
        )
      end
    end
  end
  return el
end

return {
  {Math = Math},  -- First pass: add anchors and numbers
  {Link = Link, Span = Span}  -- Second pass: fix references
}
