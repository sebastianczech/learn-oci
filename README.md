# OCI (Oracle Cloud Infrastructure)

Links, materials and notes collected while learning OCI (Oracle Cloud Infrastructure):
* [Become an OCI Generative AI Professional](https://mylearn.oracle.com/ou/learning-path/become-an-oci-generative-ai-professional/136227)
* [Become An OCI AI Foundations Associate (2023)](https://mylearn.oracle.com/ou/learning-path/become-an-oci-ai-foundations-associate-2023/127177)
* [Become An OCI Container Engine for Kubernetes Specialist](https://mylearn.oracle.com/ou/learning-path/become-an-oci-container-engine-for-kubernetes-specialist/134984)
* [Oracle Cloud Infrastructure Networking Professional](https://mylearn.oracle.com/ou/course/oracle-cloud-infrastructure-networking-professional/133455/)
* [Oracle Cloud Infrastructure Architect Associate](https://mylearn.oracle.com/ou/course/oracle-cloud-infrastructure-architect-associate/122418/)
* [OCI for AWS Architects](https://mylearn.oracle.com/ou/course/oci-for-aws-architects/104530/)
* [Free Certification for OCI Generative AI - Become an OCI Generative AI Professional - Offer Valid Now Through July 31, 2024](https://education.oracle.com/genai/?source=:so:li:or:awr:oun:::RC_WWMK240423P00002:OUOrganic&SC=:so:li:or:awr:oun:::RC_WWMK240423P00002:OUOrganic&pcode=WWMK240423P00002)
* [Generative AI](https://docs.oracle.com/en-us/iaas/Content/generative-ai/home.htm)
* [EmbedText](https://docs.oracle.com/en-us/iaas/api/#/en/generative-ai-inference/20231130/EmbedTextResult/EmbedText)
* [Oracle Communities - Generative AI](https://community.oracle.com/ou/search?domain=all_content&query=generative%20ai&scope=site&source=community)
* [What Is Generative AI (GenAI)? How Does It Work?](https://www.oracle.com/artificial-intelligence/generative-ai/what-is-generative-ai/)
* [What Is Retrieval-Augmented Generation (RAG)?](https://www.oracle.com/artificial-intelligence/generative-ai/retrieval-augmented-generation-rag/)
* [Oracle’s generative AI strategy](https://blogs.oracle.com/ai-and-datascience/post/generative-ai-strategy)
* [Oracle Cloud Infrastructure Data Science and AI services Examples](https://github.com/oracle-samples/oci-data-science-ai-samples)
* [AI Solutions Hub](https://www.oracle.com/artificial-intelligence/solutions/)
* [AI hands-on lab](https://apexapps.oracle.com/pls/apex/dbpm/r/livelabs/livelabs-workshop-cards?c=y&p100_focus_area=212:220)
* [OCI Generative AI Agents](https://www.oracle.com/artificial-intelligence/generative-ai/agents/)

## Become An OCI AI Foundations Associate (2023)

### AI foundations

* AI = artificial intelligence
  * ability of machines to mimic the cognitive abilities and problem-solving capabilities of human intelligence
  * if we can apply AGI to solve problems with specific, narrow objectives
* AGI = artificial general intelligence
  * if we can replicate human intelligence (learning skills, thinks abstractly, plan, create art) in machines
* AI examples:
  * classifying images
  * writing code
* AI tasks and data:
  * language related:
    * text-related AI tasks e.g.:
      * detect language
      * extract key phrases
    * generative AI tasks e.g.:
      * create summary
      * answer questions
    * text as data:
      * inherently sequential -> sentences
      * multiple words -> tokenization
      * varying sentence lengths -> padding
      * similar words -> dot or cosine similarity and embedding
    * language AI models:
      * designed to understand, process and generate natural language
  * speech related:
    * speech-related AI tasks e.g.:
      * turn speech to text
    * generative AI tasks e.g.:
      * music composition
    * audio as data:
      * digitized snapshots in time -> sample rate
      * sampling rate of 44.1 kHz
      * bit depth
    * audio AI models:
      * designed to manipulate audio and speech
  * vision-related:
    * image-related AI tasks e.g:
      * classify image
    * generative AI e.g.:
      * create image from text
    * image as data:
      * images consist of pixels
      * pixels are grey scale or color
    * vision AI models:
      * designed to understand information from images and videos
* AI vs ML vs DL vs GenAI
  * AI > ML > DL
  * AI - machines imitate human intelligence
  * ML - algorithms learn from past data and predict outcome on new data
    * supervised ML - learn from labeled data
    * unsupervised ML - discover trends, cluster data
    * reinforcement learning - solve problems by trial and error
  * DL - algorithms learn from complex data using neural networks and predict outcome
  * GenAI
* generative AI - ML that can produce content (audio, code, text, video, images)

### Generative AI and LLM Foundations

* GenAI - type of AI that can create new content
* subset of DL where models are trained to generate output on their own
* learns the underlying patterns in a given data set and uses that knowledge to create new data that shares those patterns
* GenAI types:
  * image-based
  * text-based
* [GAN - Generative Adversarial Network](https://www.geeksforgeeks.org/generative-adversarial-network-gan/) - generate realistic / high quality images that resemble training data
* LLMs - built to understand, generate and process human language at a massive scale
* Transformers and LLMs (Large Language Models) - based on Deep Learning architectures such as Transformers
* use cases:
  * visual (e.g. image generation)
  * language (e.g. content creation, code generation)
  * music
  * drug discovery
* LLMs features:
  * based on deep learning architecture like transformers
  * trained on vast language data
  * scaled to milions of parameters
  * enhanced contextual understanding
  * natural language understanding
* Model size and parameters
  * model size - amount of memory required to store the model's parameters
  * paremeters - numerical valuses of the model that change as it learns
  * tokens - parts of words commonly occuring sequences of characters
  * model size and number of tokens shoud be scale equally
  * scaling to larger data sets only benefical when data is high-quality
* applications of LLM:
  * text clasification
  * question answering
  * text generation
  * document summarization
* sequence model
  * solves probles where input data is in the form of sequences
  * goal - find patterns and relationships and make predicitions
  * examples:
    * NLP
    * music generation
    * speech recognition
    * gesture generation
    * time series analysis
* recurrent neural netowrks (RNN)
  * handles sequentional data
  * maintains hidden state or memory
  * allows information to persist using feedback loop
  * capture dependencies
* transformes
  * understand sentences as a whole
  * type of deep learning
  * self attention
  * encoder-decoder

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
      * e.g. when there is a significant amount of labeled data
      * e.g. when the LLM does not perform well on a task and the data for prompt engineering is too large
    * parameters efficient FT
    * soft prompting
      * e.g. when there is a need to add learnable parameters to a LLM without task-specific training
    * continual pre-training
* decoding:
  * the process of generating text with an LLM
  * happens iteratively, 1 word at a time
  * pick:
    * the highest probability word at each step (greedy decoding)
    * randomly among high probability candidates at each step (non-deterministic decoding)
  * temperature - (hyper) parameter that modulates the distribution over vocabulary
    * increasing temperature makes the model deviate more from greedy decoding
    * increasing temperature flattens the distribution, allowing for more varied word choices
* greedy decoding
  * choosing the word with the highest probability at each step of decoding
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
* challenge when applying diffusion models to text generation:
  * diffusion models work well with continuous data
  * in image generation pixel values are inherently continuous
  * text data is discrete and categorical
  * discreteness means that you can't directly apply the same gradual noise and denoising process that works for images
* fine-tunning - optimizing pretrained foundationl models on a smaller domain-specific dataset:
  * custom data + pretrained model -> fine-tunning -> custom model
  * accuracy is a performance metric that is used to evaluate classification models - it is defined as the number of correct predictions divided by the total number of predictions made
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
      * penalty function -> it penalizes a token each time it appears after the first occurence
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
      * fine-tuning typically involves adjusting the weights of a pretrained model across all layers to better perform on a specific task.
      * PEFT is a technique developed to fine-tune large models more efficiently - instead of updating all the parameters, PEFT techniques update only a small subset of the parameters or add small, trainable modules within the model while keeping most of the original pretrained parameters frozen.
    * process:
      * base-model weights + annotated training data
      * T-Few fine-tunning method
      * fine-tune weights
      * service stack (inference traffic)
    * it selectively updates only a fraction of weights to reduce computational load and avoid overfitting
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
  * unit hours required for fine-tuning:
    * cluster active 10 days -> 10 days * 24 hours * 2 units = 480 unit hours (minimum unit hours commitment is 744 unit hours for hosting commitment, not fine-tunning)
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

* Retrieval Augmented Generation (RAG):
  * methodrating text using additional information fetched from external data source
  * retrieve documents and pass then to a seq2seq model
* **RAG framework = retriever + ranker + generator**
* RAG techniques:
  * RAG sequence (like a chapter topic)
    * combines the powers of a retriever and a generator
    * for each input, it retrieves a set of relevant documents and considers them together to generate cohesive response
  * RAG token (like each sentence or even each word)
* RAG pipeline:
  * ingestion (documents -> chunks -> embedding -> index (database))
  * retrieval (query -> index -> tok K results)
  * generation (top K results -> response to user)
* RAG application:
  * prompt + chat history = enhanced prompt -> embedding model -> embedding (similarity search) -> vector ID matches -> relational database (fetch docs for matching IDs) -> augmented prompt -> LLM -> highly accurate response
* RAG evaluation:
  * RAG triad = query + context + response
  * context relevance + groundedness (factual correctness) + answer relevance (query relevance)
* Vector Databases:
  * structure of vector database vs relational databases:
    * vector db is based on distance and similarities in a vector space
  * LLM vs LLM+RAG
    * LLM = pre-training -> base LLM -> fine tunning -> fine-tunned LLM -> query / response
    * LLM+RAG = pre-training -> base LLM -> query relevant docs from Vector DB -> Q/A system -> query / response
  * Data -> Vector
  * Vector = sequence of numbers called dimensions, used to capture the important features of the data
  * Embedding in LLM is high-dimensional vector
  * Vector is generated using deep learning embedding model and represent the semantic content of data
  * Similar vectors - K-nearest neighbors algorithm (KNN)
  * Vector database workflow:
    * Vectors -> Indexing ->  Vector Database -> Querying -> Post Processing
    * indexing map vectors to a data structure for faster searching, enabling efficient retrieval
    * indexing structures organize data in a way that optimizes retrieval operations, such as nearest neighbor search, which is commonly used for finding the most similar vectors
  * cosine distance of 0 -> similar direction
  * examples of DBs:
    * [Faiss](https://github.com/facebookresearch/faiss)
    * [Pinecone](https://www.pinecone.io/)
    * [Oracle AI Vector Search](https://www.oracle.com/database/ai-vector-search/)
    * [Chroma](https://www.trychroma.com/)
    * [Weaviate](https://weaviate.io/)
  * properties of DBs:
    * accuracy
    * latency
    * scalability
  * accuracy in vector databases contributes to the effectiveness of Large Language Models (LLMs) by preserving a specific type of relationship
    * semantic relationships in vector spaces are foundational to the way LLMs understand and process language
  * role of vector DBs in LLM
    * address the hallucination (e.g. inaccuracy) problem inherent in LLM responses
    * augment prompt with enterprise-specific content to produce better responses
    * avoid exceeding LLM token limits by using most relevant content
    * cheaper than fine-tunning LLMs
    * real-time updated knowledge base
    * cache previous LLM prompts / responses to improve performance and reduce costs
* Keyword search:
  * keyword - words used to match with the terms people are searching for, when looking for products, services or general information
  * simplest form of search based on exact matches of the user provided keywords
  * evaluates documents based on the presence and frequency of the query term
* Semantic search:
  * search by meaning
  * retrieval is done by understanding intent and context, rather than matching keywords
  * ways to do it:
    * dense retrieval - uses text embeddings
    * reranking - assigns relevance score
  * embedding - represents the meaning of text as list of numbers
  * capture the essence of the data in a lower-dimensional space while maintaining the semantic relationship and meaning
  * dense retrieval:
    * relies on embeddings of both queries and documents to identify and rank relevant documents for a given query
    * enables the retrieval system to understand and match based on the contextual similarities between queries and documents
  * rerank:
    * assigns a relevance score to (query, response) pairs from initial search results
    * high relevance score pairs are more likely to be correct
    * implemented through a trained LLM
* Hybrid search -> Sparse + Dense
```
          /--> dense embedding model  --\
input data                                > hybrid -> normalization -> hybrid index
          \--> sparse embedding model --/
```

### Build an LLM application using OCI Generative AI service

* Build conversational Chatbot:
  * [LangChain](https://github.com/langchain-ai/langchain) prompts and models
  * Incorporate memory
  * Implement RAG with [LangChain](https://github.com/langchain-ai/langchain)
  * Trace LLM calls and evaluate
  * Deploy Chatbot on OCI
* ChatBot = OCI Generative AI Service + [LangChain](https://github.com/langchain-ai/langchain)
* [Setup guide](https://github.com/ou-developers/ou-generativeai-pro/blob/main/demos/module4/OU%20ChatBot%20Setup-V1.pdf)
* [Source code](https://github.com/ou-developers/ou-generativeai-pro/tree/main/demos)
* Chatbot architecture:
```
            context - document from storage
question -> prompt  --------------------------> LLM -> answer
            context - memory
```
* LangChain:
  * Python framework for developing apps powered by LLM
  * components:
    * LLM
    * prompts
    * memory
      * ability to store information about past interactions
      * chain interacts with the memory twice a run
        * after user input but before chain execution (read from memory)
        * after core logic but before output (write to memory)
      * memory per user
    * chains
    * vector stores
    * document loaders
  * models:
    * LLM
    * Chat models
  * prompt templates:
    * predefined recipes for generating prompts
  * chains:
    * using LCEL (LangChain Expression Language)
    * legacy (e.g. Python)
  * chain typically interact with memory in a run within the LangChain framework ?
    * after user input but before chain execution , and again after core logic but before output
* RAG (retrieval augmented generation) with LangChain
  * training data
  * custom data
  * the process of fetching custom information and inserting it into the model prompt is know as Retrieval Augmented Generation (RAG)
  * LLM has limited knowledge and needs to be augmented with custom data
  * indexing (local document, split documents, embed and store) + retrieval ang generation (retrieve, generate)
  * the purpose of RAG in text generation is to generate text using extra information retrieved from an external source
  * RAG combines a retrieval step with a generation step
* RAG plus Memory
  * chatbot needs to be conversational too
* Chatbot architecture:
  * indexing:
    * doc load
    * split text
    * embedding
    * Chroma DB
    * file store
  * retrieval and generation:
    * load vector store
    * DB
    * embedding
    * retriever
    * LLM
    * memory
    * chain
    * streamlit client
* Deployment to OCI:
  * VM (virtual machine)
  * source code
  * Python with virtual environment
  * dependencies
  * Chroma DB server
  * Chatbot app
* Deployment of LangChain application to Data Science as Model:
  * OCI Gen AI LLM
  * LangChain application
  * ChainDeployment class
  * Model Artifacts
  * Deployment of the model
  * Model invocation
* [Oracle Accelerated Data Science (ADS)](https://accelerated-data-science.readthedocs.io/en/latest/)
* other use case:
  * AI assisted chatbot + company policies + chat history -> model ?
  * LLM enhanced with RAG for dynamic information retrieval and response generation
* [Build a basic LLM chat app](https://docs.streamlit.io/develop/tutorials/llms/build-conversational-apps)
* [langchain.memory.buffer.ConversationBufferMemory](https://api.python.langchain.com/en/latest/memory/langchain.memory.buffer.ConversationBufferMemory.html)
* [langchain_community.chat_message_histories.streamlit.StreamlitChatMessageHistory](https://api.python.langchain.com/en/latest/chat_message_histories/langchain_community.chat_message_histories.streamlit.StreamlitChatMessageHistory.html#langchain-community-chat-message-histories-streamlit-streamlitchatmessagehistory)
  * StreamlitChatMessageHistory can be used by any type of LLM application
* string prompt templates - can support any number of variables, including none at all
  * prompt templates in language model applications typically use Python's `str.format` syntax
* loss metric - indicates how wrong the model's predictions are

### Topics to investigate

* what if cosine distance of 0 for the relationship between two embeddings ?
* when soft prompting appropriate compared to other training styles ?
* challenges to apply diffusion models to text generation ?
* fine-tunning vs parameter-efficient fine tunning (PEFT) ?
* what does the process of greedy decoding entail ?
* semantic relationship in case fo accuracy in vector DBs to the effectiveness of LLMs ?
* loss metric for model's predictions ?
* characteristics of t-few fine-tunning for LLM ?
* indexing in vector DBs ?
* what are LangChain components ?
* when fine-tunning is appropriate method for customizing LLM ?
* how concept of "Groundedness" differ from "Answer Relevance" for RAG ?
* prompt templates in Python ?
* accuracy measure for fine-tuning result ?
* evaluation of documents in keyword-based search ?
* capabilities of string prompt templates ?
* purpose of RAG in text generation ?