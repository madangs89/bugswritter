import { RecursiveCharacterTextSplitter } from "@langchain/textsplitters";

import { MemoryVectorStore } from "@langchain/classic/vectorstores/memory";
import { PDFParse } from "pdf-parse";
import mammoth from "mammoth";
import fs from "fs";
import { embeddings } from "../configs/embeding.js";
import { setVectorStore } from "../index.js";
export const splitText = async (req, res) => {
  try {
    if (!req.file && !req.body.data) {
      return res
        .status(400)
        .json({ message: "No file or text data provided.", success: false });
    }

    let text = "";
    if (req.file) {
      const { mimetype, size, path } = req.file;

      if (size > 5 * 1024 * 1024) {
        return res
          .status(400)
          .json({ message: "File size exceeds 5MB limit.", success: false });
      }

      if (mimetype === "application/pdf") {
        const buffer = fs.readFileSync(path);
        const u8Buffer = new Uint8Array(buffer);
        const parser = new PDFParse(u8Buffer);
        const result = await parser.getText();
        text = result.text;
      } else if (
        mimetype ===
        "application/vnd.openxmlformats-officedocument.wordprocessingml.document"
      ) {
        const result = await mammoth.extractRawText({ path: path });
        text = result.value;
      }

      fs.unlinkSync(req.file.path);
    } else {
      text = req.body.data;
    }

    const splitter = new RecursiveCharacterTextSplitter({
      chunkSize: 600,
      chunkOverlap: 200,
    });

    const docs = await splitter.createDocuments([text]);

    let newValue = await MemoryVectorStore.fromDocuments(docs, embeddings);
    setVectorStore(newValue);
    res.json({
      docs,
    });
  } catch (err) {
    console.error(err);
    res.status(500).json({
      message: "An error occurred while processing the request.",
      success: false,
    });
  }
};
