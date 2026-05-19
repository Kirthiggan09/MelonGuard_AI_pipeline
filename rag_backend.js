import dotenv from "dotenv";
dotenv.config();

import { DirectoryLoader } from "langchain/document_loaders/fs/directory";
import { PDFLoader } from "langchain/document_loaders/fs/pdf";

import { RecursiveCharacterTextSplitter } from "langchain/text_splitter";

import { MemoryVectorStore } from "langchain/vectorstores/memory";

import { OpenAIEmbeddings, ChatOpenAI } from "@langchain/openai";

async function main() {
  console.log("Loading PDFs...");

  // load all PDFs from data folder
  const loader = new DirectoryLoader("./data", {
    ".pdf": (path) => new PDFLoader(path),
  });

  const docs = await loader.load();

  console.log(`Loaded ${docs.length} pages`);

  // split into chunks
  const splitter = new RecursiveCharacterTextSplitter({
    chunkSize: 500,
    chunkOverlap: 100,
  });

  const splitDocs = await splitter.splitDocuments(docs);

  console.log(`Created ${splitDocs.length} chunks`);

  // embeddings
  const embeddings = new OpenAIEmbeddings({
    model: "text-embedding-3-small",
  });

  // vector store
  const vectorStore = await MemoryVectorStore.fromDocuments(
    splitDocs,
    embeddings
  );

  console.log("Vector store ready");

  // retriever
  const retriever = vectorStore.asRetriever(3);

  // LLM
  const llm = new ChatOpenAI({
    model: "gpt-4o-mini",
    temperature: 0,
  });

  // question
  const question =
    "What are the causes of powdery mildew in rockmelon plants?";

  console.log("\nQuestion:");
  console.log(question);

  // retrieve docs
  const relevantDocs = await retriever.invoke(question);

  const context = relevantDocs
    .map((doc) => doc.pageContent)
    .join("\n\n");

  // prompt
  const prompt = `
You are an agricultural assistant.

Answer ONLY using the context below.

Context:
${context}

Question:
${question}
`;

  // answer
  const response = await llm.invoke(prompt);

  console.log("\nAnswer:");
  console.log(response.content);
}

main();
