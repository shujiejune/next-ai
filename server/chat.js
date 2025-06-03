import { RecursiveCharacterTextSplitter } from "langchain/text_splitter";
import { OpenAIEmbeddings } from "langchain/embeddings/openai";
import { MemoryVectorStore } from "langchain/vectorstores/memory";
import { RetrievalQAChain } from "langchain/chains";
import { ChatOpenAI } from "langchain/chat_models/openai";
import { PromptTemplate } from "langchain/prompts";

import { PDFLoader } from "langchain/document_loaders/fs/pdf";

const chat = async (filePath = "./uploads/EE450_Project_SP24_updated.pdf", query) => {
    // step 1: load the pdf
    const loader = new PDFLoader(filePath);
    const data = await loader.load();

    // step 2: split the text into chunks
    const textSplitter = new RecursiveCharacterTextSplitter(
        { chunkSize: 500, chunkOverlap: 0 }
    );

    const splitDocs = await textSplitter.splitDocuments(data);

    // step 3: embeddings + save to vector store
    const embeddings = new OpenAIEmbeddings({
        openAIApiKey: process.env.REACT_APP_OPENAI_KEY,
    });
        
    const vectorStore = await MemoryVectorStore.fromDocuments(
        splitDocs,
        embeddings
    );

    // step 4: (optional) check the retrieved relevant documents
    const relevantDocs = await vectorStore.similaritySearch(
        "What is task decomposition?"
    );

    // step 5: quation answering
    const model = new ChatOpenAI({
        modelName: "gpt-3.5-turbo",
        openAIApiKey: process.env.REACT_APP_OPENAI_KEY,
    });
        
    const template = `Use the following pieces of context to answer the question at the end.
        If you don't know the answer, just say that you don't know, don't try to make up an answer.
        Use three sentences maximum and keep the answer as concise as possible.
        
        {context}
        Question: {question}
        Helpful Answer:`;
        
    const chain = RetrievalQAChain.fromLLM(model, vectorStore.asRetriever(), {
        prompt: PromptTemplate.fromTemplate(template),
        // returnSourceDocuments: true,
    });
        
    const response = await chain.call({
        query,
    });
        
    return response;
};

export default chat;