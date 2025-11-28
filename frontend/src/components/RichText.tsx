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
// - ## text becomes H2 headings (large, preserves **bold** inside)
// - ### text becomes H3 headings (medium, preserves **bold** inside)
// - **text:** becomes H4 headings (small, bold)
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
    // First, handle all types of links and bold text
    let parts = text.split(/(\*\*\[[^\]]+\]\(gr:[^)]+\)\*\*|\*\*\[[^\]]+\]\(am:[^)]+\)\*\*|\*\*\[[^\]]+\]\(https?:[^)]+\)\*\*|\[[^\]]+\]\([^)]+\)|\*\*[^*]+\*\*)/g);
    
    // Filter out empty strings from the split
    parts = parts.filter(part => part !== '');
    
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
      
      // Handle regular bold text (but not bold links which are handled above)
      if (part.startsWith('**') && part.endsWith('**') && !part.includes('[')) {
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
      
      return <span key={index}>{part}</span>;
    }).flat();
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
    const h2HeadingMatch = line.match(/^##\s+(.+)$/);
    const h3HeadingMatch = line.match(/^###\s+(.+)$/);
    const boldHeadingMatch = line.match(/^\*\*(.+):\*\*$/);
    
    if (h2HeadingMatch) {
      // Flush current paragraph before adding heading
      flushParagraph();
      
      // Add the H2 heading
      elements.push(
        <h2 key={elementIndex++} className="text-xl font-bold mt-6 mb-3 text-gray-800">
          {renderTextWithBold(h2HeadingMatch[1])}
        </h2>
      );
    } else if (h3HeadingMatch) {
      // Flush current paragraph before adding H3 heading
      flushParagraph();
      
      // Add the H3 heading (### headings) - preserves bold formatting inside
      elements.push(
        <h3 key={elementIndex++} className="text-lg font-semibold mt-4 mb-2 text-gray-800">
          {renderTextWithBold(h3HeadingMatch[1])}
        </h3>
      );
    } else if (boldHeadingMatch) {
      // Flush current paragraph before adding bold heading
      flushParagraph();
      
      // Add the bold heading (for ISSUE/S, SC RULING, etc.)
      elements.push(
        <h4 key={elementIndex++} className="text-base font-bold mt-4 mb-2 text-gray-800">
          {boldHeadingMatch[1]}
        </h4>
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