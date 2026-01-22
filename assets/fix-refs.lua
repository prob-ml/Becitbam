-- Lua filter to fix LaTeX equation references for HTML output
-- Converts unresolved [eq:label] references to proper numbered links

local equation_numbers = {}
local current_eq = 0

-- First pass: collect equation labels and assign numbers
function Math(el)
  if el.mathtype == "DisplayMath" then
    local label = el.text:match("\\label{([^}]+)}")
    if label then
      current_eq = current_eq + 1
      equation_numbers[label] = current_eq
    end
  end
  return el
end

-- Second pass: fix references in links
function Link(el)
  local target = el.target
  if target:match("^#eq:") then
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
  {Math = Math},  -- First pass
  {Link = Link, Span = Span}  -- Second pass
}
