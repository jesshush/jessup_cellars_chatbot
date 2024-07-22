## Jessup_Cellars_Chatbot
### Problem Statement:
Corpus about the information about a business who sells wines. They have their own website which customers often visit. 
Taking this into concern, the business owners want to deploy a chatbot on their website. They want the chatbot to be able to answer the questions from this corpus and not use the other information. If the user asks anything that is not there in the corpus, it should just tell them to contact the business directly.

### Task :
1. Minimalistic UI - Where users can chat with the chatbot. 
2. Informed Responses - The chatbot answers from the corpus given. 
3. Out-of-Corpus Questions - And for any out of corpus questions, it tells the users to contact the business directly.
4. Minimum Latency - A maximum latency of 2-3 seconds.
5. Context Memory - Chatbot maintains the conversation history.

<br /> 

> **Optional** - Set up a virtual environment (using `GIT`) 

```bash
$ python -m venv venv
$ source venv/bin/activate  # On Windows use `venv\Scripts\activate`
```

> **Step 1** - Clone the code from the repository (using `GIT`) 

```bash
$ git clone https://github.com/jesshush/jessup_cellars_chatbot.git
$ cd jessup_cellars_chatbot
```
<br /> 

> **Step 2** - Install the libs through terminal

```bash
$ pip install -r requirements.txt
```
<br /> 

> **Step 3** - Run the app 


```bash
$ python train.py   # enter quit to end chat
$ python chat.py
$ python app.py
```
<br /> 

 > **Step 4** - Follow the local address link and click on the chat icon 


PDF - [Open DOCX](https://docs.google.com/document/d/1wSuQIET0yOqcntjAnYaocc9rHXbUbQzus8p1Wog0_J0/edit?usp=sharing)

Demo Video - [Watch the video](https://www.youtube.com/watch?v=OUeXnvitgkU)
