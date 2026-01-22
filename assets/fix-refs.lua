-- Lua filter to fix LaTeX references for HTML output
-- 1. Adds id anchors to equations with \label
-- 2. Adds equation numbers (1), (2), etc. to labeled equations
-- 3. Converts unresolved references to proper numbered links
-- 4. Ensures theorem references are just numbers, not "Theorem X"

local equation_numbers = {}
local theorem_numbers = {}
local figure_numbers = {}
local current_eq = 0
local current_thm = 0
local current_fig = 0

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

-- Track theorem numbers from Div elements with theorem class
function Div(el)
  if el.classes:includes("theorem") then
    current_thm = current_thm + 1
    if el.identifier and el.identifier ~= "" then
      theorem_numbers[el.identifier] = current_thm
    end
  end
  if el.classes:includes("figure") then
    current_fig = current_fig + 1
    if el.identifier and el.identifier ~= "" then
      figure_numbers[el.identifier] = current_fig
    end
  end
  return el
end

-- Second pass: fix references in links
function Link(el)
  local target = el.target
  local label = target:sub(2)  -- Remove the leading #
  
  if target:match("^#eq:") then
    local num = equation_numbers[label]
    if num then
      el.content = {pandoc.Str("(" .. num .. ")")}
    end
  elseif target:match("^#thm:") then
    local num = theorem_numbers[label]
    if num then
      -- Include "Theorem" prefix for theorem references
      el.content = {pandoc.Str("Theorem " .. tostring(num))}
    end
  elseif target:match("^#fig:") then
    -- For figure references, prepend "Figure " to existing content
    -- Get existing text content
    local existing_text = pandoc.utils.stringify(el.content)
    if existing_text and existing_text ~= "" then
      el.content = {pandoc.Str("Figure " .. existing_text)}
    end
  end
  return el
end

-- Also handle Span elements which may contain unresolved references
function Span(el)
  if el.attributes["data-reference-type"] == "ref" then
    local label = el.attributes["data-reference"]
    if label then
      -- Check equations
      local num = equation_numbers[label]
      if num then
        return pandoc.Link(
          {pandoc.Str("(" .. num .. ")")},
          "#" .. label
        )
      end
      -- Check theorems - include "Theorem" prefix
      num = theorem_numbers[label]
      if num then
        return pandoc.Link(
          {pandoc.Str("Theorem " .. tostring(num))},
          "#" .. label
        )
      end
      -- Check figures - include "Figure" prefix
      if label:match("^fig:") then
        local existing_text = pandoc.utils.stringify(el.content)
        if existing_text and existing_text ~= "" then
          return pandoc.Link(
            {pandoc.Str("Figure " .. existing_text)},
            "#" .. label
          )
        end
      end
    end
  end
  return el
end

return {
  {Div = Div, Math = Math},  -- First pass: collect numbers and add equation anchors
  {Link = Link, Span = Span}  -- Second pass: fix references
}
