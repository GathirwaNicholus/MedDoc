This a machine learning powered Medical assistant.
Takes natural language as the input then processes it passes it to MedDoc, ml model, (the 
processing is also done by an ml en_core_web_sm to recognize similar terms) then MedDoc outputs
whether one has heart disease or not. The output plus the prompt is passed to another model
, mixtral that is a chatbot for recommendations to the patient. There is also a BERT-Model
for classification as the Model classifies the patients as either having heart disease or
not.
All the data ie input, preprocessed_input to MedDoc, response from MedDoc and the Mixtral model
are all saved in the conversations.txt file which when missing is created a fresh.
Remember everything runs locally.

There are some missing files:
ModelFile- This is the saved status or the trained MedDoc model, if this is missing training of 
the model begins and will be recreated (training the model takes some time.).
capybarahermes-2.5-mistral-7b.Q5_K_M.gguf - This is the mixtral model (about 5GB in size.). This
model has been used as it uses relatively small resources, has a high accuracy and has relatively
fast responses compared to the very highly accurate and unbiased dolphin-2.5-mixtral-8x7b.Q5_K_M_2.gguf
model( recommended.), requires atleast 38gb of ram.
{the models can be downloaded from hugging face from theBloke - a user who has trained said 
models. Mixtral is from a russian company}

Remaining things to implement:
Sending the tokens to the webpage as they are produced (does so in the terminal.) to give it
an illusion of fast responses (about 1.7 sec per token for the capybarahermes model and 5 sec 
per token for the dolphin variant but about 2 to 3 minutes max for the total message for the capybarahemes
model and for the dolphin variant even longer.
The unique id on the landing pages seems to be consistent, not changing per session (used the uuid
package for generation of the unique ids to maintain anonymity of users and users can save 
that uuid if they would like to view their data stored later.

Note: The model only allows input when cerain keyword are input otherwise the model produces
a dissmissive message : "Hello, While that might be interesting, what specific heart health questions do you have? ".
The keyword are ["heart", "health", "angina", "cholesterol", "age"], more can be added or that
logic can be removed from the CModel.py files.

Running of the model: Run the CModel.py file as it imports from MedDoc file and uses the templates
login.html first which is just a landing page, then to the generate.html page.
