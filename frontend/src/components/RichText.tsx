"use client";


type Props = {
  content: string;
  className?: string;
};

// Minimal renderer for plain-text with basic structure:
// - Paragraphs split by blank lines
// - Lines starting with "- " become bullet items
// - Single newlines preserved within paragraphs
export function RichText({ content, className }: Props) {
  const paragraphs = (content || "").split(/\n\n+/);

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
                <li key={i}>{line.replace(/^\s*-\s+/, "")}</li>
              ))}
            </ul>
          );
        }

        // Not a pure list: render as a paragraph preserving soft line breaks
        return (
          <p key={idx} className="whitespace-pre-line">
            {para}
          </p>
        );
      })}
    </div>
  );
}

export default RichText;


