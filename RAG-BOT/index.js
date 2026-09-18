export let vectorStore = null;

export const setVectorStore = (store) => {
  vectorStore = store;
};

export const getVectorStore = () => vectorStore;
