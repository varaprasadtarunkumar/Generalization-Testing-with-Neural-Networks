# Generalization of Transformer Model on COGS, SCAN, and PCFG Datasets

During my internship at IIIT Hyderabad, I worked on testing how well the Transformer model can generalize using datasets such as COGS, SCAN, and PCFG. These datasets evaluate a model's ability to understand compositional generalization and language structure in various formats.

## Datasets

### 1. COGS (Compositional Generalization)
The COGS dataset is designed to test compositional generalization in natural language understanding tasks. The dataset consists of sentences transformed into logical forms. It focuses on how well models generalize to novel combinations of concepts seen during training.

- **Example:**
  - Input: "The researcher in a room froze."
  - Output: `researcher ( x _ 1 ) ; researcher . nmod . in ( x _ 1 , x _ 4 ) AND room ( x _ 4 ) AND freeze . theme ( x _ 5 , x _ 1 )`

### 2. SCAN (Compositional Navigation Task)
SCAN consists of a set of commands and their corresponding action sequences. These are actions that an agent should perform to execute the commands successfully. SCAN was developed to test whether models can generalize to new combinations of instructions.

- **Examples:**
  - Input: "jump"
  - Output: `JUMP`
  
  - Input: "jump left"
  - Output: `LTURN JUMP`

  - Input: "jump thrice"
  - Output: `JUMP JUMP JUMP`

  - Input: "jump opposite left after walk around left"
  - Output: `LTURN WALK LTURN WALK LTURN WALK LTURN WALK LTURN LTURN JUMP`

### 3. PCFG (Probabilistic Context-Free Grammar)
PCFG is a dataset used to assess generalization of grammar-based tasks. It leverages probabilistic context-free grammar to generate structured sentences that test a model's ability to capture grammatical rules.

---

## Example Table

Here is an example table showing dataset types, inputs, outputs, and models tested on them:

![Example Table](https://path_to_your_image/Screenshot%202024-06-25%20141407.png)

| Dataset | Data Type                         | Input Example                                         | Output Example                                          | Models Tested                            |
|---------|-----------------------------------|-------------------------------------------------------|---------------------------------------------------------|------------------------------------------|
| SCAN    | Command to action sequence        | "jump twice"                                          | "JUMP JUMP"                                              | LSTM, GRU, Transformer                   |
| COGS    | Sentence to logical form          | "The dog chased the cat."                             | `chase(agent=dog, patient=cat)`                         | Transformer, BERT, GPT, T5, LSTM         |
| CFQ     | Natural language to SPARQL query  | "Who directed the movie starring Tom Hanks?"          | `SELECT ?x WHERE { ?m rdf:type :Film . ?m :starring :Tom_Hanks . ?m :director ?x . }` | Transformer, BERT, T5, GNN, Seq2Seq     |

---

## Insights from the Internship

This project involved evaluating how well the Transformer model could generalize to new examples and unseen combinations in each dataset. For example, SCAN tests whether the Transformer can handle action sequences formed by new combinations of instructions, while COGS examines its ability to transform sentences into logical forms with unseen combinations.

