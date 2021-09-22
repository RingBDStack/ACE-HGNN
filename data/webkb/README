This directory contains a selection of the WebKB dataset (http://www-2.cs.cmu.edu/~webkb/).

In this corpus, we extracted all the webpages except for the webpages belonging to the "other" class. These webpages are classified into one of the following five classes:
			course
			faculty
			student
			project
			staff

The webpages were gathered from four different universities and we have preserved that format. There are two files corresponding to each university: a .content file and a .cites file.

After stemming and removing stopwords we were left with a vocabulary of size 1703 unique words. All words with document frequency less than 10 were removed.


DESCRIPTION OF THE FILES:

The .content file contains descriptions of the webpages in the following format:
		<webpage_id> <word_attributes>+ <class_label>

The first entry in each line contains the unique string ID of the webpage followed by binary values indicating whether each word in the vocabulary is present (indicated by 1) or absent (indicated by 0) in the webpage. Finally, the last entry in the line contains the class label of the webpage.

The .cites file contains the citation graph of the corpus. Each line describes a link in the following format:

		<ID of cited webpage> <ID of citing webpage>

Each line contains two webpage IDs. The first entry is the ID of the webpage being cited (head of the link) and the second ID stands for the webpage which contains the citation (tail of the link). The direction of the link is from right to left. If a line is represented by "webpage1 webpage2" then the link is "webpage2->webpage1". 