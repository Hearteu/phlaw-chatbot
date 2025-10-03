"use client";


type Props = {
  content: string;
  className?: string;
  onCaseNumberClick?: (caseNumber: string, caseType: 'gr' | 'am') => void;
};

// Enhanced renderer for text with markdown-like formatting:
// - Paragraphs split by blank lines
// - Lines starting with "- " become bullet items
// - **text** becomes bold text
// - Single newlines preserved within paragraphs
// - [text](gr:123) and [text](am:123) become clickable case number links
export function RichText({ content, className, onCaseNumberClick }: Props) {
  const paragraphs = (content || "").split(/\n\n+/);

  // Function to render text with bold formatting and clickable links
  const renderTextWithBold = (text: string) => {
    // First, handle clickable links with bold formatting: **[text](gr:123)**
    let parts = text.split(/(\*\*\[[^\]]+\]\(gr:[^)]+\)\*\*|\*\*\[[^\]]+\]\(am:[^)]+\)\*\*)/g);
    
    // Then handle regular bold text within each part
    parts = parts.map(part => {
      // Handle clickable case number links
      const grMatch = part.match(/\*\*\[([^\]]+)\]\(gr:([^)]+)\)\*\*/);
      const amMatch = part.match(/\*\*\[([^\]]+)\]\(am:([^)]+)\)\*\*/);
      
      if (grMatch || amMatch) {
        return part; // Return as-is for clickable links (handled below)
      }
      
      // Handle regular bold text
      return part.split(/(\*\*[^*]+\*\*)/g);
    }).flat();
    
    return parts.map((part, index) => {
      // Handle clickable case number links
      const grMatch = part.match(/\*\*\[([^\]]+)\]\(gr:([^)]+)\)\*\*/);
      const amMatch = part.match(/\*\*\[([^\]]+)\]\(am:([^)]+)\)\*\*/);
      
      if (grMatch) {
        const [, linkText, caseNumber] = grMatch;
        return (
          <strong key={index}>
            <button
              onClick={() => onCaseNumberClick?.(caseNumber, 'gr')}
              className="text-blue-600 hover:text-blue-800 hover:underline cursor-pointer bg-transparent border-none p-0 font-bold"
              title={`View case digest for ${linkText}`}
            >
              {linkText}
            </button>
          </strong>
        );
      }
      
      if (amMatch) {
        const [, linkText, caseNumber] = amMatch;
        return (
          <strong key={index}>
            <button
              onClick={() => onCaseNumberClick?.(caseNumber, 'am')}
              className="text-blue-600 hover:text-blue-800 hover:underline cursor-pointer bg-transparent border-none p-0 font-bold"
              title={`View case digest for ${linkText}`}
            >
              {linkText}
            </button>
          </strong>
        );
      }
      
      // Handle regular bold text
      if (part.startsWith('**') && part.endsWith('**')) {
        return <strong key={index}>{part.slice(2, -2)}</strong>;
      }
      
      return part;
    });
  };

  return (
    <div className={className}>
      {paragraphs.map((para, idx) => {
        const lines = para.split(/\n/);
        const bulletLines = lines.filter((l) => /^\s*-\s+/.test(l));
        const isList = bulletLines.length > 0 && bulletLines.length === lines.length;

        if (isList) {
          return (
            <ul key={idx} className="list-disc pl-5 space-y-1">
              {lines.map((line, i) => (
                <li key={i}>{renderTextWithBold(line.replace(/^\s*-\s+/, ""))}</li>
              ))}
            </ul>
          );
        }

        // Not a pure list: render as a paragraph preserving soft line breaks
        return (
          <p key={idx} className="whitespace-pre-line">
            {renderTextWithBold(para)}
          </p>
        );
      })}
    </div>
  );
}

export default RichText;


