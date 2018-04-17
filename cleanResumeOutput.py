# Sanitize the resume output
import os, re

resumedir = 'data/pqpResumes/'

resumes = []
for filename in os.listdir(resumedir):
    resumeObject = open(resumedir + filename, 'r', errors='replace')
    resume = resumeObject.read()
    resume = resume[2:-2]
    newfile = open('data/cleanResumes/'+filename, "a+")
    newfile.write(resume)
    newfile.close()
