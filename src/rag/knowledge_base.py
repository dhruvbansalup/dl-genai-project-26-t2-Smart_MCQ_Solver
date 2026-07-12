class MCQKnowledgeBase:
    def __init__(self):
        self.documents = []

    def __len__(self):
        return len(self.documents)

    def __getitem__(self, index):
        return self.documents[index]

    def clear(self):
        self.documents = []

    def build(self, df):
        """
        Building Knowledge Base from the dataframe
        """

        self.clear()

        for _, row in df.iterrows():
            correct_option = row['answer']
            doc=str(f"Prompt: {row['prompt']}\n"
                    f"Answer: {row[correct_option]}")
            self.documents.append(doc)
        return self

    def get_documents(self):
        """
        Return complete document list
        """
        return self.documents