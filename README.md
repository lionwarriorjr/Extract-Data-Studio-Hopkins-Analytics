# Extract-Data-Studio-Hopkins-Analytics
Source code for knowledge discovery engine developed at Hopkins for the facilitation of complex queries and visualizations using natural language in sports analytics.

## Oracle API example
username = USERNAME
api_key = API_KEY
desc_filename = os.path.join(os.getcwd(), 'dataset/sample_query.csv')
config = Config(username, api_key, desc_filename) # construct Config object with customizable domain knowledge
qry = "what is the strike rate of ranger pitchers on fastballs or curveballs against lefty batters on pitches with the start velocity over        90 in the eighth inning"
oracle = Oracle(config) # construct instance of Oracle engine
result, sample_size, plot, rterms = oracle.run(qry) # return result, sample size, visualization, and relevance feedback
