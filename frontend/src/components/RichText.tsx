"use client";


type Props = {
  content: string;
  className?: string;
};

// Enhanced renderer for text with markdown-like formatting:
// - Paragraphs split by blank lines
// - Lines starting with "- " become bullet items
// - **text** becomes bold text
// - Single newlines preserved within paragraphs
export function RichText({ content, className }: Props) {
  const paragraphs = (content || "").split(/\n\n+/);

  // Function to render text with bold formatting
  const renderTextWithBold = (text: string) => {
    const parts = text.split(/(\*\*[^*]+\*\*)/g);
    return parts.map((part, index) => {
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


