export const config = {
  runtime: 'edge',
};

// Token from environment variable
const DEFAULT_HF_TOKEN = process.env.HF_TOKEN || '';

const SYSTEM_INSTRUCTION = 
  "You are NLRC AI, an advanced frontier-grade multimodal reasoning assistant developed with custom GRPO neural logic.\n" +
  "Your name is strictly NLRC AI. Never identify as Qwen, DeepSeek, Llama, or any other base model.\n\n" +
  "Multimodal & Vision Capabilities:\n" +
  "- You possess genuine visual perception. When an image is provided, thoroughly examine its visual details, layout, text/OCR, subjects, colors, UI elements, and spatial relationships.\n" +
  "- Break down screenshots, charts, diagrams, and photos with deep analytical clarity.\n\n" +
  "Real-Time World Knowledge & Dispatches:\n" +
  "- When [Verified Real-Time World Dispatches] are included in your context, treat them as confirmed ground-truth facts from the live internet.\n" +
  "- Synthesize breaking news, current affairs, and developments from the past 24-48 hours accurately with clear citations.\n\n" +
  "Reasoning Guidelines:\n" +
  "- In <think>...</think> tags, plan your approach, verify facts, calculate intermediate steps, inspect visual elements or live dispatches, and test edge cases. Conclude with </think>.\n\n" +
  "Final Answer Guidelines (MANDATORY):\n" +
  "- After </think>, you MUST deliver an exhaustive, highly detailed, and complete explanation.\n" +
  "- NEVER give a 1-2 sentence summary. The final answer must contain:\n" +
  "  1. Detailed step-by-step analysis, proofs, or exhaustive visual scene breakdown.\n" +
  "  2. Clear Markdown headings (###), bullet points, and comparative tables where helpful.\n" +
  "  3. Complete code implementations with thorough explanations and complexity analysis (if technical).\n" +
  "  4. Rich contextual explanations so the user understands the *why* and *how*, not just the result.";

async function fetchLiveNews(query) {
  try {
    const clean = query
      .replace(/in\s+the\s+last\s+\d+\s+hours|what\s+are\s+the|tell\s+me\s+about|please\s+give\s+me|can\s+you\s+tell\s+me|what\s+is\s+happening\s+in/gi, '')
      .replace(/[^\w\s]/gi, ' ')
      .trim();
    const searchQuery = clean.length >= 3 ? clean : query.trim();

    const url = `https://news.google.com/rss/search?q=${encodeURIComponent(searchQuery)}&hl=en-IN&gl=IN&ceid=IN:en`;
    const res = await fetch(url, { headers: { 'User-Agent': 'Mozilla/5.0' } });
    if (!res.ok) return [];
    const xml = await res.text();

    const items = [];
    const itemMatches = xml.match(/<item>([\s\S]*?)<\/item>/g) || [];
    for (const itemXml of itemMatches.slice(0, 5)) {
      const titleMatch = itemXml.match(/<title>(.*?)<\/title>/);
      const pubDateMatch = itemXml.match(/<pubDate>(.*?)<\/pubDate>/);
      const sourceMatch = itemXml.match(/<source[^>]*>(.*?)<\/source>/);
      if (titleMatch) {
        const title = titleMatch[1]
          .replace(/<!\[CDATA\[(.*?)\]\]>/g, '$1')
          .replace(/&amp;/g, '&')
          .replace(/&#39;/g, "'")
          .replace(/&quot;/g, '"')
          .trim();
        const date = pubDateMatch ? pubDateMatch[1].trim() : '';
        const source = sourceMatch ? sourceMatch[1].trim() : 'Verified News';
        items.push({ title, date, source });
      }
    }
    return items;
  } catch (err) {
    console.warn('Live news retrieval error:', err);
    return [];
  }
}

export default async function handler(req) {
  // Handle CORS preflight
  if (req.method === 'OPTIONS') {
    return new Response(null, {
      status: 204,
      headers: {
        'Access-Control-Allow-Origin': '*',
        'Access-Control-Allow-Methods': 'GET, POST, OPTIONS',
        'Access-Control-Allow-Headers': 'Content-Type, Authorization',
      },
    });
  }

  if (req.method !== 'POST') {
    return new Response(JSON.stringify({ error: 'Method not allowed' }), {
      status: 405,
      headers: { 'Content-Type': 'application/json', 'Access-Control-Allow-Origin': '*' },
    });
  }

  try {
    const body = await req.json();
    const { messages = [], max_tokens = 1800, model: requestedModel, webSearch } = body;

    const token = process.env.HF_TOKEN || DEFAULT_HF_TOKEN;

    // Detect if multimodal vision image is present
    let hasImage = false;
    for (const m of messages) {
      if (Array.isArray(m.content)) {
        if (m.content.some(c => c.type === 'image_url' || c.image_url)) {
          hasImage = true;
          break;
        }
      } else if (m.image) {
        hasImage = true;
        break;
      }
    }

    // Build payload messages with proper OpenAI multimodal format
    const payloadMessages = messages.map(m => {
      if (m.image && !Array.isArray(m.content)) {
        return {
          role: m.role,
          content: [
            { type: 'text', text: m.content || 'Please analyze this image in detail.' },
            { type: 'image_url', image_url: { url: m.image } }
          ]
        };
      }
      return m;
    });

    if (payloadMessages.length === 0 || payloadMessages[0].role !== 'system') {
      payloadMessages.unshift({ role: 'system', content: SYSTEM_INSTRUCTION });
    }

    // Real-Time Web News Grounding
    if (!hasImage) {
      const lastUserMsg = [...payloadMessages].reverse().find(m => m.role === 'user');
      let lastUserText = '';
      if (lastUserMsg) {
        if (typeof lastUserMsg.content === 'string') {
          lastUserText = lastUserMsg.content;
        } else if (Array.isArray(lastUserMsg.content)) {
          const textPart = lastUserMsg.content.find(c => c.type === 'text');
          if (textPart) lastUserText = textPart.text;
        }
      }

      const isNewsQuery = /(last\s+24|last\s+48|past\s+24|past\s+48|current\s+affairs|today|yesterday|breaking\s+news|latest\s+news|recent\s+news|who\s+won|election|news\s+in)/i.test(lastUserText);

      if ((isNewsQuery || webSearch) && lastUserText) {
        const liveItems = await fetchLiveNews(lastUserText);
        if (liveItems.length > 0) {
          const now = new Date();
          const dateStr = now.toLocaleDateString('en-US', { weekday: 'long', year: 'numeric', month: 'long', day: 'numeric' });
          const dispatches = liveItems.map(it => `• [${it.source} | ${it.date}]: ${it.title}`).join('\n');
          const contextInjection = `[Verified Real-Time World Dispatches — ${dateStr}]:\n${dispatches}\n\n`;

          if (typeof lastUserMsg.content === 'string') {
            lastUserMsg.content = contextInjection + `User Inquiry: ${lastUserMsg.content}`;
          } else if (Array.isArray(lastUserMsg.content)) {
            const textPart = lastUserMsg.content.find(c => c.type === 'text');
            if (textPart) {
              textPart.text = contextInjection + `User Inquiry: ${textPart.text}`;
            }
          }
        }
      }
    }

    // Dedicated Endpoint for Our Actual Trained 8B Model (Google Cloud GPU)
    const NLRC_8B_ENDPOINT = process.env.NLRC_8B_ENDPOINT || 'https://products-most-plastics-nick.trycloudflare.com/v1/chat/completions';

    if (hasImage) {
      // Vision queries use the dedicated multimodal vision engine
      const visionRes = await fetch('https://router.huggingface.co/featherless-ai/v1/chat/completions', {
        method: 'POST',
        headers: {
          'Authorization': `Bearer ${token}`,
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          model: 'Qwen/Qwen3-VL-30B-A3B-Instruct',
          messages: payloadMessages,
          max_tokens,
          stream: true,
        }),
      });

      if (visionRes.ok) {
        return new Response(visionRes.body, {
          status: 200,
          headers: {
            'Content-Type': 'text/event-stream',
            'Cache-Control': 'no-cache, no-transform',
            'Connection': 'keep-alive',
            'Access-Control-Allow-Origin': '*',
            'X-NLRC-Model': 'NLRC-AI-Vision-30B',
          },
        });
      }
    }

    // Text queries route EXCLUSIVELY to our actual trained 8B model weights
    try {
      const hfRes = await fetch(NLRC_8B_ENDPOINT, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          model: 'NLRC-AI-Reasoning-8B',
          messages: payloadMessages,
          max_tokens,
          stream: true,
        }),
      });

      if (hfRes.ok) {
        return new Response(hfRes.body, {
          status: 200,
          headers: {
            'Content-Type': 'text/event-stream',
            'Cache-Control': 'no-cache, no-transform',
            'Connection': 'keep-alive',
            'Access-Control-Allow-Origin': '*',
            'X-NLRC-Model': 'NLRC-AI-Reasoning-8B (Trained Weights)',
          },
        });
      } else {
        const errText = await hfRes.text();
        return new Response(JSON.stringify({ 
          error: 'NLRC AI 8B GPU server returned an error: ' + errText 
        }), {
          status: 502,
          headers: { 'Content-Type': 'application/json', 'Access-Control-Allow-Origin': '*' },
        });
      }
    } catch (err) {
      return new Response(JSON.stringify({ 
        error: 'Your trained NLRC AI 8B GPU server is currently offline or unreachable. Please launch the GPU server on Colab to connect your weights.',
        details: err.message
      }), {
        status: 503,
        headers: { 'Content-Type': 'application/json', 'Access-Control-Allow-Origin': '*' },
      });
    }

    return new Response(JSON.stringify({ error: 'All Hugging Face GPU endpoints are currently unavailable.' }), {
      status: 502,
      headers: { 'Content-Type': 'application/json', 'Access-Control-Allow-Origin': '*' },
    });
  } catch (err) {
    return new Response(JSON.stringify({ error: err.message || 'Internal server error' }), {
      status: 500,
      headers: { 'Content-Type': 'application/json', 'Access-Control-Allow-Origin': '*' },
    });
  }
}
