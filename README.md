# Extract-Data-Studio-Hopkins-Analytics
Source code for knowledge discovery engine developed at Hopkins for the facilitation of complex queries and visualizations using natural language in sports analytics.

## Oracle API example
username = USERNAME  
api_key = API_KEY <br /><br />
desc_filename = os.path.join(os.getcwd(), 'dataset/sample_query.csv')<br />
config = Config(username, api_key, desc_filename) # construct Config object with customizable domain knowledge <br /><br />
qry = "what is the strike rate of ranger pitchers on fastballs or curveballs against lefty batters on pitches with the <br />
       start velocity over 90 in the eighth inning" <br /><br />
oracle = Oracle(config) # construct instance of Oracle engine <br /><br />
result, sample_size, plot, rterms = oracle.run(qry) # return result, sample size, visualization, and relevance feedback

## How to query the Oracle API:
domain_knowledge.py and the modules subdirectory provide the terms in domain knowledge that can be queried against (namely by specifying the feature => feature descriptor mappings). Each module comes with a set of words that describe how to call it.

Note: Almost all queries handled by the system currently are of the "What is" Jeopardy form, i.e. "what is the XXX of YYY batters on ZZZ pitches ...".

Data: Domain Knowledge currently supports the MLB Statcast dataset.

## Flashcard Code Example
[flashcard={batter=Jose Altuve}]
Code for querying on flashcard generation currently takes a different form, as described above. Here we mark the module to select (flashcard), an equals (=) sign, followed by a dictionary of the arguments to the module (here batter=Jose Altuve). This is a form that we are adopting for all modules that take parameters as arguments. Filters can be added on top of this (such as querying to filter in a particular inning, etc.).
