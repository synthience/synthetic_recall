# Techniques

## Tree of Thoughts (ToT)

### Introduction

For complex tasks that require exploration or strategic planning, traditional prompting techniques often fall short. To address these limitations, researchers **Yao et al. (2023)** and **Long (2023)** proposed the **Tree of Thoughts (ToT)** framework. ToT generalizes chain-of-thought prompting by encouraging exploration over multiple reasoning paths—referred to as "thoughts"—that serve as intermediate steps toward solving a problem with language models (LMs).

### How Tree of Thoughts Works

The ToT framework maintains a **tree structure of thoughts**, where each node represents a coherent language sequence acting as an intermediate step in problem-solving. This approach allows a language model to:

- **Generate Thoughts**: Produce potential reasoning steps toward a solution.
- **Evaluate Thoughts**: Assess the viability of each thought using the model's internal knowledge and reasoning abilities.
- **Search Through Thoughts**: Employ search algorithms like **breadth-first search (BFS)** and **depth-first search (DFS)** to systematically explore different reasoning paths, enabling lookahead and backtracking.

By integrating these components, ToT enables the model to consider multiple potential solutions before arriving at a final answer, enhancing its problem-solving capabilities.

### Implementation Details

When applying ToT, specific parameters must be defined based on the task:

- **Number of Candidates (b)**: The number of thought candidates to retain at each step.
- **Number of Steps (T)**: The depth to which the reasoning tree is expanded.

#### Example: Game of 24

In the **Game of 24**, players use four numbers and arithmetic operations to reach the number 24. This task requires decomposing the problem into a series of intermediate equations.

- **Process**:
  - **Step 1**: Generate initial thoughts (possible equations) using the given numbers.
  - **Step 2**: Evaluate each thought as "sure," "maybe," or "impossible" regarding its potential to reach 24.
  - **Step 3**: Use BFS or DFS to explore the most promising thoughts, keeping the best **b = 5** candidates at each step.

- **Evaluation Criteria**:
  - **Promote Correct Solutions**: Identify partial solutions likely to lead to 24.
  - **Eliminate Improbable Paths**: Discard thoughts that are logically impossible.
  - **Retain Uncertain Paths**: Keep "maybe" thoughts for further exploration.

Values are sampled multiple times for each thought to enhance diversity and robustness.

### Results and Performance

Experimental results indicate that ToT significantly outperforms other prompting methods on complex reasoning tasks:

- **Higher Accuracy**: In tasks like the Game of 24, ToT achieved superior accuracy compared to standard chain-of-thought prompting and other baselines.
- **Efficient Problem Solving**: By systematically exploring multiple reasoning paths, ToT reduces the likelihood of getting stuck in incorrect or suboptimal solutions.

**Note**: For detailed charts and figures illustrating these results, please refer to the original research papers.

### Variations and Related Work

#### Reinforcement Learning-Based ToT

At a high level, ToT shares similarities with other approaches that enhance LLMs' problem-solving abilities through tree search:

- **Yao et al. (2023)**: Utilize standard search algorithms like BFS, DFS, and beam search without task-specific adaptations.
- **Long (2023)**: Introduce a "ToT Controller" trained via reinforcement learning (RL) to guide the search process. This controller learns optimal backtracking and exploration strategies, allowing the system to evolve and acquire new knowledge even with a fixed underlying LLM.

#### Tree-of-Thought Prompting

**Hulbert (2023)** proposed **Tree-of-Thought Prompting**, applying the main concepts of ToT as a straightforward prompting technique:

- **Approach**: Encourage the LLM to evaluate intermediate thoughts within a single prompt.
- **Sample Prompt**:

  ```
  Imagine three different experts are answering this question.
  All experts will write down one step of their thinking, then share it with the group.
  Then all experts will proceed to the next step, and so on.
  If any expert realizes they're wrong at any point, they will leave.
  The question is...
  ```

#### PanelGPT

**Sun (2023)** conducted large-scale experiments on Tree-of-Thought Prompting and introduced **PanelGPT**:

- **Concept**: Simulate a panel discussion among multiple instances of LLMs.
- **Benefit**: Leverages diverse perspectives and collaborative reasoning to enhance problem-solving.
- **Outcome**: Demonstrated improved performance on tasks requiring complex reasoning.

### Advantages of Tree of Thoughts

- **Enhanced Exploration**: Systematically explores multiple reasoning paths, reducing the chance of overlooking correct solutions.
- **Improved Accuracy**: By evaluating and pruning thoughts, the model avoids pursuing unpromising paths.
- **Dynamic Reasoning**: Allows for lookahead and backtracking, mimicking human strategic thinking.

### Applications

- **Mathematical Problem Solving**: Excels in tasks that require multi-step calculations and reasoning.
- **Logical Reasoning Tasks**: Suitable for puzzles and games that involve strategic planning.
- **Complex Decision-Making**: Can be applied to scenarios where multiple factors must be considered and weighed.

### Conclusion

The Tree of Thoughts framework represents a significant advancement in leveraging language models for complex problem-solving tasks. By maintaining a tree of reasoning paths and employing search algorithms for systematic exploration, ToT enables models to consider a broader range of potential solutions and improves their ability to arrive at correct answers.

As research continues, variations like RL-based controllers and collaborative prompting techniques (e.g., PanelGPT) offer promising directions for further enhancing the reasoning capabilities of language models.

---

**References**

- **Yao, S., et al. (2023)**. *Tree of Thoughts: Deliberate Problem Solving with Large Language Models*. [arXiv:2305.10601](https://arxiv.org/abs/2305.10601)
- **Long, G. (2023)**. *Enhancing Language Model Reasoning with Reinforcement Learning-Based Tree of Thoughts*. [Link to paper if available]
- **Hulbert, A. (2023)**. *Tree-of-Thought Prompting for Language Models*. [Blog Post or Article Link]
- **Sun, X. (2023)**. *PanelGPT: Prompting with Panel Discussions among Language Models*. [Link to paper if available]

**Code Repository**

- [Tree of Thoughts Implementation](https://github.com/ysymyth/tree-of-thought-llm)

---

This enhanced explanation provides a clear and organized overview of the Tree of Thoughts technique, improving readability and comprehension while retaining the essential details of the original content.