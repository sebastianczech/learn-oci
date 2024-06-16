# OCI (Oracle Cloud Infrastructure)

Links, materials and notes collected while learning OCI (Oracle Cloud Infrastructure):
* [Become an OCI Generative AI Professional](https://mylearn.oracle.com/ou/learning-path/become-an-oci-generative-ai-professional/136227)
* [Become An OCI Container Engine for Kubernetes Specialist](https://mylearn.oracle.com/ou/learning-path/become-an-oci-container-engine-for-kubernetes-specialist/134984)
* [Oracle Cloud Infrastructure Networking Professional](https://mylearn.oracle.com/ou/course/oracle-cloud-infrastructure-networking-professional/133455/)
* [Oracle Cloud Infrastructure Architect Associate](https://mylearn.oracle.com/ou/course/oracle-cloud-infrastructure-architect-associate/122418/)
* [OCI for AWS Architects](https://mylearn.oracle.com/ou/course/oci-for-aws-architects/104530/)

## Become an OCI Generative AI Professional

### Fundamentals of LLM

Notes:
* LLM = Large Language Models
* **LLM is a probabilistic model of text**
* architectures:
  * encoder - model that convert a sequence of words to an embedding (vector representation) (e.g. embedding tokens)
  * decoder - models take a sequence of words and output next word (e.g text generation)
  * encoder-decoder - encodes a sequence of words and use the encoding + output a next word
* transformers = encoder + decoder
* prompt - the text provided to an LLM as input, sometimes containing instruction and/or examples
* prompting -  the process of providing an initial input or question to a LLM to guide its generation of text or to elicit a specific type of response.
* prompt engineering - the process of iteratively refining a prompt for the purpose of eliciting a particular style of response
* k-shot prompting - explicitly providing *k* examples of the intended task in the prompt
* prompting strategies:
  * chain-of-thought - prompt the LLM to emit intermediate reasoning steps
  * least-to-most - prompt the LLM to decompose the problem and solve, easy first
  * step-back - prompt the LLM to identify high-level concepts pertinent to a specific task
* issues:
  * prompt injection (jailbreaking) - to deliberately provide an LLM with input that attempts to cause it to ignore instructions, cause harm, behave contrary to deployment expectations
  * memorization (after answering repeat the prompt)
    * leaked prompt
    * leaked private information
* training:
  * prompting alone may be inappropriate when:
    * training data exists
    * domain adoption is required
  * domain-adaption - adapting a model to enhance its performance outside of the domain it was trained on
  * styles:
    * fine-tuning (FT)
    * parameters efficient FT
    * soft prompting
    * continual pre-training
* decoding:
  * the process of generating text with an LLM
  * happens iteratively, 1 word at a time
  * pick:
    * the highest probability word at each step (greedy decoding)
    * randomly among high probability candidates at each step (non-deterministic decoding)
  * temperature - (hyper) parameter that modulates the distribution over vocabulary
    * increasing temperature makes the model deviate more from greedy decoding
* hallucination:
  * generated text that is non-factual and/or ungrounded
  * how to reduce it ? e.g. retrieval-augmentation
  * there is no known methodology to reliable keep LLMs from hallucinating
* groundedness and attributability:
  * grounded - generated text is grounded in a document if the document supports text
* LLM applications:
  * retrieval augmented generation (RAG)
  * code models
  * multi-model
  * language agents

Links:
* [Encoder and Decode explained with practical example](https://medium.com/@muhammad.a0625/encoder-and-decode-explained-with-practical-example-20d19bfe77e9)
* [Understanding Encoder And Decoder LLMs](https://magazine.sebastianraschka.com/p/understanding-encoder-and-decoder)
* [Encoder-Decoder Models for Natural Language Processing](https://www.baeldung.com/cs/nlp-encoder-decoder-models)
* [Encoder Decoder Architecture](https://www.larksuite.com/en_us/topics/ai-glossary/encoder-decoder-architecture)

### Generative AI service

* OCI generative AI service:
  * fully managed service that provides set of customizable LLMs available via API (endpoints)
  * choice of models
  * flexible fine-tunning
  * dedicated AI clusters (GPU based, dedicated RDMA cluster network)
* text input -> OCI generative AI service -> text output
* pretrained foundational models
  * text generation:
    * command (from cohere)
    * command-light (from cohere)
    * 11ama 2-70b-chat (from meta)
  * text summarization:
    * command
  * embedding (convert text to vector embeddings, semantic search, multilingual models):
    * embed-english-v3.0
    * embed-multilingual-v3.0
* fine-tunning - optimizing pretrained foundationl models on a smaller domain-specific dataset:
  * custom data + pretrained model -> fine-tunning -> custom model
* generation models:
  * token - part of a word, entire word or punctuation
  * use cases - text generation, chat, text summarization
  * parameters in OCI:
    * maximum output tokens
    * temperature - a (hyper) parameter that controls the randomness of the LLM output
      * 0 - model is deterministic
      * increased -> distribution is flattened over all words (model uses words with lower probabilities)
    * top p, top k
      * top k - tells model to pick the next token from the top k tokens in its list, sorted by probability
      * top p - picks from the top tokens based on the sum of their probabilities
    * presence / frequency penalty - useful when you want to get rid of repetition in your outputs
    * stop sequence - string that tells the model to stop generating more content
    * show likelihoods - every time a new token is generated, a number between -15 and 0 is assigned to all tokens
* summarization models
  * generates a succinct version of the original text that relays the most important information
  * parameters:
    * temperature
    * length of the summary (short, medium, long)
    * format (free form or in bullet points)
    * extractiveness (how much to reuse the input)
* embedding models
  * embeddings - numerical representation of a piece of text converted to number sequences
  * word embedding - capture properties of the word
  * actual embedding represents more properties (coordinates) that just 2
  * rows of coordinates are called vectors and represented as numbers
  * semantic similarity - embeddings that are numerically similar, are also semantically similar
  * sentence embedding - associates every sentence with a vector of numbers
  * similar sentences are assigned to similar vectors
* prompt engineering and LLM customization
  * prompt - input or initial text provided to the model
  * prompt engineering - the process of iteratively refining a prompt for the purpose of eliciting a particular style of response
  * prompt ---input---> LLM
  * generated text <---output--- LLM
  * LLM as next work predictors
  * aligning LLM to follow instructions
  * reinforce learning from human feedback (RLHF) to fine-tune LLMs to follow broad class of written instructions
  * in-context learning and few-shot prompting
    * in-context learning - conditioning (prompting) an LLM with instructions and of demonstrations of the task it is meant to complete
    * k-shot prompting - explicitly providing k examples of the intended task in the prompt
  * prompt formats
    * [Llama2 prompt formatting](https://llama.meta.com/docs/model-cards-and-prompt-formats/meta-llama-2/)
```
<<s>
[INST]
  <<SYS>>
  {{message}}
  <</SYS>>
[/INST]
```
  * advanced prompting strategies
    * chain-of-thought - provide examples in a prompt
    * zero shot chain-of-thought
* customize LLM with your data
  * why not to do it:
    * expensive (cost)
    * a lot of data needed
    * expertise (thorough understanding of model performance)
  * 3 methods:
    * in-context learning / few shot prompting
    * fine-tunning
    * RAG
* retrieval augmented generation (RAG)
  * language model is able to query enterprise knowledge bases to provide grounded responses
  * RAG do not require custom models
* fine-tunning and inference
  * optimize a model on a smaller domain-specific dataset
  * model is fine-tuned by taking a pretrained foundational model and providing additional training using custom data
  * benefits:
    * improve model performance on specific tasks
    * improve model efficiency (reduced number of tokens)
  * custom model - model created using a pretrained model as a base and using own dataset to fine-tune that model
  * steps:
    * create a dedicated AI cluster
    * gather training data
    * kickstart fine-tuning
    * fine-tuned (custom) model generated
  * model endpoint - designated point on a dedicated AI cluster where large language model can accept user requests and send back responses such as the model's generated text
  * T-Few fine-tuning
    * selectively updates only a fraction of the model's weights
    * additive Few-Shot Parameter Efficient Fine Tuning (PEFT)
    * process:
      * base-model weights + annotated training data
      * T-Few fine-tunning method
      * fine-tune weights
      * service stack (inference traffic)
  * reduce inference costs
    * computationally expensive
    * share GPU resources
    * running within dedicated RDMA network
  * inference serving with minimal overhead
     * GPU memory is limited so switching between models can incur significant overhead due to reloading the full GPU memory
     * minimal overhead when switching between models derived from the same base model
  * configuration
    * training methods:
      * Vanilla
      * T-Few
    * hyperparamenters:
      * total training epochs (T-Few)
      * learning rate (T-Few)
      * training batch size (T-Few)
      * early stopping patience (T-Few)
      * early stopping threshold (T-Few)
      * log model metrics interval in steps (T-Few)
      * number of last layers (Vanilla)
    * results
      * accuracy
      * loss
* dedicated AI clusters
  * effectively a single-tenant deployment where the GPUs in the cluster only host custom models
  * types:
    * fine-tuning
    * hosting (for inference)
  * unit size:
    * large cohere (for hosting or fine-tunning)
    * small cohere
    * embed cohere (for hosting)
    * llama2-70 (for hosting Llama2 models)
  * capabilities:
    * text generation
    * summarization
    * embedding
  * fine-tunning requires more GPUs that hosting a model
  * same cluster can host up to 50 different fine-tuned models
  * pricing:
    * commitments:
      * min hosting commitment - 744 unit-hours/cluster
      * min fine-tunning commitment - 1 unit-hour/fine-tunning job
    * unit hours - each fine-tunning cluster requires 2 units and each cluster is active for 5 hours
    * fine-tuning or hosting cost / month
* generative AI security architecture
  * security and privacy of customer workloads is an essential design tenet
  * GPUs allocated for a customer's generative AI tasks are isolated from other GPUs
  * dedicated RDMA network with GPU pool -> allocated into dedicated AI cluster
  * for strong data privacy, dedicated GPU cluster only handles fine-tuned models of a single customer
  * base model + fine-tuned model endpoints share the same cluster responsible for the most efficient utilization of GPUs in dedicated AI cluster
  * customer data access is restricted withing customer's tenancy
  * leverage OCI security services:
    * authentication and authorization (IAM)
    * key management
    * object storage buckets (encrypted by default)

### Building blocks for an LLM application

* Retrieval Augmented Generation (RAG)
  * method of generating text using additional information fetched from external data source
  * retrieve documents and pass then to a seq2seq model
* **RAG framework = retriever + ranker + generator**
* RAG techniques:
  * RAG sequence (like a chapter topic)
  * RAG token (like each sentence or even each word)
* RAG pipeline:
  * ingestion (documents -> chunks -> embedding -> index (database))
  * retrieval (query -> index -> tok K results)
  * generation (top K results -> response to user)
* RAG application:
  * prompt + chat history = enhanced prompt -> embedding model -> embedding (similarity search) -> vector ID matches -> relational database (fetch docs for matching IDs) -> augmented prompt -> LLM -> highly accurate response
* RAG evaluation:
  * RAG triad = query + context + response
  * context relevance + groundedness + answer relevance

### Build an LLM application using OCI Generative AI service