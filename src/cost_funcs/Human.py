import numpy as np
import random
import pandas as pd

# Human (agent) is defined as class
# each individual agent have several properties


class Human:
    def __init__(self, uid):
        self.uid = uid  # id
        self.houseX = None  # household x-coordinate
        self.houseY = None  # household y-coordinate
        self.schoolID = None  # school ID
        self.schoolX = None  # school x-coordinate
        self.schoolY = None  # school y-coordinate
        self.workID = None  # workplace ID
        self.workX = None  # work x-coordinate
        self.workY = None  # work y-coordinate
        self.currentX = None  # current x-coordinate of agent
        self.currentY = None  # current y-coordinate of agent
        self.infX = None  # x coordinate where agent gets exposed
        self.infY = None  # y coordinate where agent gets exposed
        self.infT = None  # when agent gets exposed
        self.S = True  # susceptible
        self.E = False  # exposed
        self.I = False  # infectious
        self.R = False  # recovered
        self.contactList = None  # people who are collocated

    def infecting(self, peopleList, infRate, reproduction, time):
        # reproduction number is drawn from possion distribution
        reproductionCount = np.random.poisson(reproduction)
        for i in range(0, reproductionCount):
            if (self.I == True) & (random.random() <= infRate):
                randomValue = random.random()
                if randomValue < 2/3:  # people staying at their home for 16 hours
                    contactList = [peopleList[i] for i in range(
                        len(peopleList)) if self.houseID == peopleList[i].houseID]
                    person = random.choice(contactList)
                    if (person.S == True):
                        person.S = False
                        person.E = True
                        person.infX = person.houseX
                        person.infY = person.houseY
                        person.infT = time
                else:  # people staying at their workplace/schools for 8 hours
                    if (self.age >= 6 & self.age <= 19):  # students
                        contactList = [peopleList[i] for i in range(
                            len(peopleList)) if self.schoolID == peopleList[i].schoolID]
                        person = random.choice(contactList)
                        if (person.S == True):
                            person.S = False
                            person.E = True
                            person.infX = person.schoolX
                            person.infY = person.schoolY
                            person.infT = time
                    if (self.age >= 20 & self.age <= 65):  # workers
                        contactList = [peopleList[i] for i in range(
                            len(peopleList)) if self.workID == peopleList[i].workID]
                        person = random.choice(contactList)
                        if (person.S == True):
                            person.S = False
                            person.E = True
                            person.infX = person.workX
                            person.infY = person.workY
                            person.infT = time

    def incubating(self):
        if (self.E == True):
            if (random.random() <= 1/3):  # 3 days for incubation periods
                self.E = False
                self.I = True

    def recovering(self):
        if (self.I == True):
            if (random.random() <= 1/7):  # 7 days for infectious periods
                self.I = False
                self.R = True


# set the age distribution, which should be similar to the ACS data
def settingHumanAgent(houseList):
    totalPop = np.sum(
        [houseList[i].houseMemberCount for i in range(len(houseList))])
    weightedRange = random.choices([range(0, 5), range(5, 10),
                                    range(10, 15), range(15, 20),
                                    range(20, 25), range(25, 30),
                                    range(30, 35), range(35, 40),
                                    range(40, 45), range(45, 50),
                                    range(50, 55), range(55, 60),
                                    range(60, 65), range(65, 70),
                                    range(70, 75), range(75, 80),
                                    range(80, 85), range(85, 90)],
                                   weights=[0.058, 0.056, 0.056, 0.058, 0.067, 0.073, 0.070, 0.069, 0.073, 0.074, 0.075,
                                            0.063, 0.055, 0.045, 0.037, 0.028, 0.021, 0.022],
                                   k=totalPop)

    members = pd.DataFrame({'members': weightedRange})
    popMembers = pd.DataFrame(members['members'].value_counts())
    popMembers['index'] = popMembers.index.values
    agePop = []

    for i in range(0, 18):
        age_range = popMembers['index'][i]
        counts = popMembers['members'][i]
        age = random.choices(age_range, k=counts)
        agePop.append(age)
    agePop = np.hstack(agePop)
    random.shuffle(agePop)

    households = []
    for i in range(len(houseList)):
        house = houseList[i]
        for j in range(house.houseMemberCount):
            households.append(house)

    peopleList = [Human(i) for i in range(len(agePop))]

    for i in range(len(peopleList)):
        person = peopleList[i]
        person.age = agePop[i]
        person.houseID = households[i].houseID
        person.houseX = households[i].houseX
        person.houseY = households[i].houseY
        person.currentX = person.houseX
        person.currentY = person.houseY
        person.schoolID = households[i].schoolID
        person.schoolX = households[i].schoolX
        person.schoolY = households[i].schoolY
        person.workID = households[i].workID
        person.workX = households[i].workX
        person.workY = households[i].workY

    return(peopleList)
