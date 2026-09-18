import { ai } from "../configs/gemini.js";
import { getVectorStore } from "../index.js";

export const handleChat = async (req, res) => {
  try {
    const { query, history } = req.body;
    const vectorStore = getVectorStore();

    let sources = [];

    if (!vectorStore) {
      return res.status(400).json({
        message: "Vector store not initialized. Upload a document first.",
        success: false,
      });
    }

    const response = await vectorStore.similaritySearchWithScore(query, 3);


    const filtered = response.slice(0, 3);

    if (filtered.length === 0) {
      return res.json({
        answer: "It is not clearly defined in the document.",
        success: true,
        sources,
      });
    }

    const context = filtered.map(([doc]) => doc.pageContent).join("\n\n");

    sources = filtered.map(([doc, score]) => ({
      content: doc.pageContent.slice(0, 200),
      score,
    }));

    if (!context.trim()) {
      return res.json({
        answer: "It is not clearly defined in the document.",
        success: true,
        sources,
      });
    }

    const chat = ai.chats.create({
      model: "gemini-3-flash-preview",
      history: history || [],
      config: {
        systemInstruction: `
You are a document-based question answering assistant.

You MUST follow these rules strictly:

1. Answer ONLY using the provided context.
2. Do NOT use any external knowledge.
3. Do NOT guess or assume anything.
4. If the answer is not clearly found in the context, you MUST respond with:
"It is not clearly defined in the document."
5. If the context does not contain enough information, DO NOT attempt to answer.
6. Always return your response in STRICT JSON format like this:

{
  "answer": "your answer here"
}

7. Do NOT return anything outside JSON.
`,
        responseJsonSchema: {
          type: "object",
          properties: {
            answer: {
              type: "string",
            },
          },
        },
      },
    });

    const response1 = await chat.sendMessage({
      message: `
Context:
${context}

Question:
${query}
`,
    });

    let parsedAnswer;

    try {
      parsedAnswer = JSON.parse(response1.text);
    } catch (e) {
      parsedAnswer = {
        answer: "It is not clearly defined in the document.",
      };
    }

    return res.json({
      answer: parsedAnswer.answer,
      success: true,
      sources,
    });
  } catch (err) {
    console.error("Error in handleChat:", err);
    return res.status(500).json({
      message: "An error occurred while processing the request.",
      success: false,
    });
  }
};
