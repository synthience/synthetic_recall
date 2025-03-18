# Techniques

## Prompt Chaining

### Introduction to Prompt Chaining

Prompt chaining is a prompt engineering technique used to enhance the reliability and performance of Large Language Models (LLMs). It involves breaking down complex tasks into smaller, manageable subtasks. Once these subtasks are identified, the LLM is prompted with each subtask sequentially, and the response from one subtask serves as input to the next. This process creates a chain of prompts, hence the term **prompt chaining**.

Prompt chaining is particularly useful for accomplishing complex tasks that an LLM might struggle to address when given a single, detailed prompt. By decomposing the task, the model can focus on one aspect at a time, improving overall performance. Additionally, prompt chaining allows for transformations or additional processing on generated responses before reaching the final desired output.

This technique not only boosts the performance of LLMs but also enhances the transparency, controllability, and reliability of your application. By isolating each step, you can more easily debug issues, analyze performance at different stages, and make targeted improvements where necessary. Prompt chaining is especially beneficial when building LLM-powered conversational assistants and enhancing personalization and user experience in applications.

### Use Cases for Prompt Chaining

#### Prompt Chaining for Document Question Answering

Prompt chaining can be applied in various scenarios that involve multiple operations or transformations. A common use case is answering questions about a large text document. Instead of prompting the LLM with a single, complex question, you can design two distinct prompts:

1. **Prompt 1**: Extract relevant quotes from the document based on the question.
2. **Prompt 2**: Use the extracted quotes and the original document to compose a detailed answer to the question.

By creating two separate prompts, you enable the LLM to focus on specific tasks sequentially, improving accuracy and reliability.

**Example:**

Suppose you have a large document and a question you want to answer. Here's how you can implement prompt chaining:

**Prompt 1: Extract Relevant Quotes**

```
You are a helpful assistant. Your task is to help answer a question given in a document. The first step is to extract quotes relevant to the question from the document, delimited by ####. Please output the list of quotes using <quotes></quotes>. Respond with "No relevant quotes found!" if no relevant quotes were found.

####
{{document}}
####

Question: {{question}}
```

*Note: For testing purposes, you can use an article from Wikipedia, such as the page on [Prompt Engineering](https://en.wikipedia.org/wiki/Prompt_engineering).*

**Output of Prompt 1:**

```html
<quotes>
- Chain-of-thought (CoT) prompting
- Generated knowledge prompting
- Least-to-most prompting
- Self-consistency decoding
- Complexity-based prompting
- Self-refine
- Tree-of-thought prompting
- Maieutic prompting
- Directional-stimulus prompting
- Textual inversion and embeddings
- Using gradient descent to search for prompts
- Prompt injection
</quotes>
```

**Prompt 2: Compose an Answer Using Extracted Quotes**

```
Given a set of relevant quotes (delimited by <quotes></quotes>) extracted from a document and the original document (delimited by ####), please compose an answer to the question. Ensure that the answer is accurate, has a friendly tone, and sounds helpful.

<quotes>
{{extracted_quotes}}
</quotes>

####
{{document}}
####

Question: {{question}}
```

**Output of Prompt 2:**

The prompting techniques mentioned in the document include:

1. Chain-of-thought (CoT) prompting
2. Generated knowledge prompting
3. Least-to-most prompting
4. Self-consistency decoding
5. Complexity-based prompting
6. Self-refine
7. Tree-of-thought prompting
8. Maieutic prompting
9. Directional-stimulus prompting
10. Textual inversion and embeddings
11. Using gradient descent to search for prompts
12. Prompt injection

Each of these techniques employs unique strategies to enhance or specify interactions with large language models to produce the desired outcomes.

**Note:** You may want to clean up the quotes further by removing any unnecessary citations or formatting issues before using them in the second prompt.

### Benefits of Prompt Chaining

- **Improved Performance**: By breaking down complex tasks, the LLM can provide more accurate and coherent responses.
- **Enhanced Transparency**: Each step in the chain can be individually reviewed and assessed.
- **Greater Controllability**: Allows for fine-tuning and adjustments at each stage of the process.
- **Ease of Debugging**: Simplifies the identification and correction of errors in the model's responses.

### Additional Resources

For more examples of prompt chaining, you can refer to [Anthropic's documentation on prompt design](https://docs.anthropic.com/claude/docs/prompt-design), which leverages the Claude LLM. Our example was inspired and adapted from their examples.

### Further Techniques

- **Generate Knowledge Prompting**: Enhancing LLM responses by generating additional knowledge or context before answering the main question.
- **Tree of Thoughts**: A method where the LLM explores multiple reasoning paths (thoughts) in a tree-like structure to arrive at the best answer.

---

This revised explanation provides a clearer, more organized overview of prompt chaining, enhancing readability and understanding while maintaining the original content.