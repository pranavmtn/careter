"""Personal names Careter never suggests (it still treats them as correctly spelled).
Names that are also everyday words (will, mark, grace, rose, hope, may, june, frank, bill, carol,
dean, drew, miles, rich, art, guy, ray, sue, bob, ken, lee, ivy, ruby, pearl, maya...) are
deliberately left out, so those words still get suggested."""

NAMES = frozenset("""
aaron abigail adam adrian aiden alan albert alex alexander alexis alice alicia allison amanda amber amy
andrea andrew angela ann anna anne anthony antonio ashley audrey austin ava barbara benjamin betty beverly
billy brandon brenda brian brittany bruce bryan caleb cameron carl carlos caroline catherine charles
charlotte cheryl chloe christina christine christopher cindy claire cynthia daniel danielle david debbie
deborah debra denise dennis diana diane donald donna doris dorothy douglas dylan edward elijah elizabeth
ella emily emma eric ethan eugene evelyn gabriel gary george gerald gloria gregory hannah harold heather
helen henry isabella jacob jacqueline james janet janice jason jayden jeffrey jennifer jeremy jerry jesse
jessica joan john johnny jonathan jordan jose joseph joshua joyce juan judith judy julia julie justin
karen katherine kathleen kathryn kayla keith kelly kenneth kevin kimberly kyle larry laura lauren liam
linda lisa logan lori louis lucas madison margaret maria marie marilyn martha mary matthew megan melissa
michael michelle mia natalie nathan nicholas nicole noah olivia oliver pamela patricia patrick paul peter
philip rachel ralph randy raymond rebecca richard robert roger ronald russell ruth ryan samantha samuel
sandra sara sarah scott sean sharon shirley sophia sophie stephanie stephen steven susan teresa theresa
thomas timothy tyler victoria vincent walter wayne william zachary zoe michel jessie kate ben tim jim
dan phil joe tom steve mike dave chris matt josh nick jenny kathy

johnson williams jones garcia miller davis rodriguez martinez hernandez lopez gonzalez wilson anderson
taylor moore jackson martin thompson harris clark lewis robinson perez allen sanchez scott torres nguyen
flores adams nelson mitchell roberts gomez phillips evans diaz parker cruz edwards collins reyes stewart
morris morales murphy rogers gutierrez ortiz morgan peterson kelly howard ramos richardson watson chavez
bennett mendoza ruiz hughes alvarez castillo myers ross jimenez smith brown davies wright

rahul priya anjali arjun amit ankit aditya akshay rohit sachin sanjay suresh ramesh rajesh mahesh deepak
deepa divya sneha pooja neha kavya meera anu anil sunil manoj vinod pranav abhishek abhijith akhil nikhil
arun varun kiran gokul hari jithin jishnu sreejith sreekumar unni midhun vipin vivek sajith renjith
rejith shyam ajay vijay jayan sujith sreeja reshma aswathy athira anjana sruthi gopika nimisha nithya
parvathy revathi devika keerthi lekha aparna fathima ayesha muhammed mohammed mohammad shibu biju saji
shaji raju ravi manu anoop anand aravind hrishikesh adarsh aswin ashwin nandana nandu sharma singh kumar
gupta nair menon pillai kurup varma iyer reddy rao patel shah mehta ghosh bose mukherjee chatterjee
banerjee joshi kulkarni deshmukh verma mishra pandey tiwari yadav agarwal jain chopra kapoor malhotra
hussain ahmed varghese cherian kurian mathew abraham jacobs jomon jibin jinu linu sinu tinu anish
""".split())


def is_name(word):
    return word.lower() in NAMES
