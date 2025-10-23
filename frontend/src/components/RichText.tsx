"use client";


type Props = {
  content: string;
  className?: string;
  onCaseNumberClick?: (caseNumber: string, caseType: 'gr' | 'am') => void;
};

// Enhanced renderer for text with markdown-like formatting:
// - Paragraphs split by blank lines
// - Lines starting with "- " or "* " become bullet items
// - **text** becomes bold text
// - ## text becomes headings
// - Single newlines preserved within paragraphs
// - [text](gr:123) and [text](am:123) become clickable case number links
// - [text](url) becomes clickable external links
export function RichText({ content, className, onCaseNumberClick }: Props) {
  // Split content into lines and process each line
  const lines = (content || "").split(/\n/);
  const elements: React.ReactElement[] = [];
  let currentParagraph: string[] = [];
  let elementIndex = 0;

  // Function to render text with bold formatting and clickable links
  const renderTextWithBold = (text: string) => {
    // First, handle clickable links with bold formatting: **[text](gr:123)**
    let parts = text.split(/(\*\*\[[^\]]+\]\(gr:[^)]+\)\*\*|\*\*\[[^\]]+\]\(am:[^)]+\)\*\*|\*\*\[[^\]]+\]\(https?:[^)]+\)\*\*)/g);
    
    // Then handle regular bold text and markdown links within each part
    parts = parts.map(part => {
      // Handle clickable case number links
      const grMatch = part.match(/\*\*\[([^\]]+)\]\(gr:([^)]+)\)\*\*/);
      const amMatch = part.match(/\*\*\[([^\]]+)\]\(am:([^)]+)\)\*\*/);
      const urlMatch = part.match(/\*\*\[([^\]]+)\]\((https?:[^)]+)\)\*\*/);
      
      if (grMatch || amMatch || urlMatch) {
        return part; // Return as-is for clickable links (handled below)
      }
      
      // Handle regular bold text and markdown links
      return part.split(/(\*\*[^*]+\*\*|\[[^\]]+\]\([^)]+\))/g);
    }).flat();
    
    return parts.map((part, index) => {
      // Handle clickable case number links
      const grMatch = part.match(/\*\*\[([^\]]+)\]\(gr:([^)]+)\)\*\*/);
      const amMatch = part.match(/\*\*\[([^\]]+)\]\(am:([^)]+)\)\*\*/);
      const urlMatch = part.match(/\*\*\[([^\]]+)\]\((https?:[^)]+)\)\*\*/);
      
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
      
      if (urlMatch) {
        const [, linkText, url] = urlMatch;
        return (
          <strong key={index}>
            <a
              href={url}
              target="_blank"
              rel="noopener noreferrer"
              className="text-blue-600 hover:text-blue-800 hover:underline cursor-pointer font-bold"
              title={`Open ${linkText} in new tab`}
            >
              {linkText}
            </a>
          </strong>
        );
      }
      
      // Handle regular bold text
      if (part.startsWith('**') && part.endsWith('**')) {
        return <strong key={index}>{part.slice(2, -2)}</strong>;
      }
      
      // Handle regular markdown links [text](url)
      const markdownLinkMatch = part.match(/\[([^\]]+)\]\(([^)]+)\)/);
      if (markdownLinkMatch) {
        const [, linkText, url] = markdownLinkMatch;
        return (
          <a
            key={index}
            href={url}
            target="_blank"
            rel="noopener noreferrer"
            className="text-blue-600 hover:text-blue-800 hover:underline cursor-pointer"
            title={`Open ${linkText} in new tab`}
          >
            {linkText}
          </a>
        );
      }
      
      return part;
    });
  };

  // Process lines
  const flushParagraph = () => {
    if (currentParagraph.length > 0) {
      const paragraphText = currentParagraph.join('\n');
      
      // Check if this is a bullet list (support both - and * bullets)
      const bulletLines = currentParagraph.filter(line => /^\s*[-*]\s+/.test(line));
      const isList = bulletLines.length > 0 && bulletLines.length === currentParagraph.length;
      
      if (isList) {
        elements.push(
          <ul key={elementIndex++} className="list-disc pl-5 space-y-1">
            {currentParagraph.map((line, i) => (
              <li key={i}>{renderTextWithBold(line.replace(/^\s*[-*]\s+/, ""))}</li>
            ))}
          </ul>
        );
      } else {
        elements.push(
          <p key={elementIndex++} className="whitespace-pre-line">
            {renderTextWithBold(paragraphText)}
          </p>
        );
      }
      currentParagraph = [];
    }
  };

  for (const line of lines) {
    // Check if this line is a heading
    const headingMatch = line.match(/^##\s+(.+)$/);
    const boldHeadingMatch = line.match(/^\*\*(.+):\*\*$/);
    
    if (headingMatch) {
      // Flush current paragraph before adding heading
      flushParagraph();
      
      // Add the heading
      elements.push(
        <h2 key={elementIndex++} className="text-xl font-bold mt-6 mb-3 text-gray-800">
          {headingMatch[1]}
        </h2>
      );
    } else if (boldHeadingMatch) {
      // Flush current paragraph before adding bold heading
      flushParagraph();
      
      // Add the bold heading (for ISSUE/S, SC RULING, etc.)
      elements.push(
        <h3 key={elementIndex++} className="text-lg font-bold mt-4 mb-2 text-gray-800">
          {boldHeadingMatch[1]}
        </h3>
      );
    } else if (line.trim() === '') {
      // Empty line - flush current paragraph
      flushParagraph();
    } else {
      // Regular line - add to current paragraph
      currentParagraph.push(line);
    }
  }

  // Flush any remaining paragraph
  flushParagraph();

  return (
    <div className={className}>
      {elements}
    </div>
  );
}

export default RichText;