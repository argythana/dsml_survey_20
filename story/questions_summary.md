## According to the Manual, the Survey essentially contains ten parts.

To the extend that questions may overlap, they may be used to assess the quality of the answer, whether this is intentional by design or not.

### 1st part: general demographic (sample identification)(Q1-Q6).
    In questionnaire surveys, such questions help identify the characteristics on the population that may explain other features. Primary demographic variables may provide info about secondary explanatory variables (could be examined from a dependent and explanatory point of view) and variables that we might want to treat as dependent variables.
    
    Q1: Age
    Q2: Gender
    Q3: country of residence
    Q4: Education level: Have or plan => two questions in one
    Q5: role
    Q6: Coding/programming exp
 **Remark: Q15 and Q6 are the two "Experience questions". Can be combined/compared, used for data validation. **

### 2nd part contains 2 sub parts, software - hardware:
a) 6 general questions about software tools:

    Q7: programming language use, all
    Q8: programming language recommend to learn first, One
    Q9: IDE use, all
    Q10: hosted notebook products use, all
        Important note: Not all free -> we gain different info from different categories.
    Q14: Data Visualization Libraries, all
    
    + extra question:
    Q38: primary tool to analyze data in work or school ONE, (including "Other" in text)

and 3 general questions about hardware
    
    Q11: Computing platform most often, One
    Q12: types of specialized hardware, all
    Q13: How often special Hardware (TPU). 


### 3rd part focuses on ML.
Remark: Q15 and Q6 are the two "Experience questions". Can be combined/compared, used for data validation.
    
    Q15: Years of using ML methods (may group with demographic vars and coding exp.)
    Q16: is about ML libraries which may be also grouped together with programming software questions.
    Q17: is about ML Algos in general
        Q18: is about Computer vision (subset of Q17)
        Q19: is about NLP methods (subset of Q17)

### 4rth part is about Company of employment and work demographics:
    
    Q20: Company size
    Q21: Company Data Science team size
    Q22: Company and ML methods employment. Can be validated, combined, compared with Q25, Q28-A and 34-A, 35-A
    Q23: Job role

### 5th part is about compensation and cost:
    
    Q24: Your compensation = employers cost.
    Q25: ML and cloud computing services cost.


*After Q25 (cloud costs) => from this part and on, many questions are split to branches, different for "Pros and non-pros"*
    
    Non-professionals were defined as:
    students, unemployed, and respondents that have never spent any money in the cloud.
    Is the distinction well founded in the data?
    e.g. Someone self-employed, with dedicated workstation and his own server (or company servers) might be as pro as others.
    We have to check the replies of zero costs joined with experience, role, position, hardware used, salary, and what else?

Those are questions of special interest to cloud service providers besides users.

### 6th part is about cloud services:
    
    Q26-A: cloud computing platforms do you use regularly, all
        Q27-A: cloud computing products REGULARLY (subset of Q26A), all
        Q28-A: ML products REGULARLY (subset of Q26A), all
    Q26-B: cloud computing platforms to learn in next 2 years?, ONE
        Q27-Β: cloud computing products to learn in next 2 years?, all
        Q28-Β: ML products REGULARLY to learn in next 2 years?, all
        
### 7th part is about databases, big data and similar:
    
    Q29-A: big data products regularly, all
        Q30: big data product Most often (subset of Q29-A). Exact same choices, One
    Q29-B: big data products to learn in 2 years, all    
    
### 8th part is about Business Intelligence tools:
     
     Q31-A: Which tools regularly, all
         Q32: Which tool most often? Exact same choices, ONE
     Q31-B: Which tools to learn in next 2 years, all

### 9th part is about automated ML tools and products:
    Can be cross-validated, compared, combined with Q22, Q25, Q28-A and 34-A, 35-A
    
    Q33-A: categories of auto ML tools, all
    Q33-B: categories of auto ML tools to learn in next 2 years, all
        Q34-A: specific auto ML tools for each category, all (subset of Q33-A)
        Q34-B: specific auto ML tools for each category, all (subset of Q33-B)
     
    Q35-A: ML managing tools, all
    Q35-B: ML managing tools, to learn in next 2 years, all

### 10th part is about MLDS social "media":
    
    Q36: Where do you share or deploy?
    Q37: Where do you learn, educate yourself?
    Q39: Social media sources for DSML
