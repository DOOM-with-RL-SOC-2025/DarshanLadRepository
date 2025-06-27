"""
We'll try to understand classes in python. 
Check the resources on google classroom to ensure you have gone through everything expected.

"""
###### THESE LISTS HAVE ALREADY BEEN DEFINED FOR YOU ###############
engineer_roster = [] # A list of all instantiated engineer objects
sales_roster = [] # List of all instantiated sales objects
branchmap = {  # A dictionary of dictionaries -> Maps branchcodes to cities and branch names
    0:  { "city": "NYC", "name": "Hudson Yards"},
    1:  { "city": "NYC" , "name": "Silicon Alley"},
    2:  { "city": "Mumbai", "name": "BKC"},
    3:  { "city": "Tokyo", "name": "Shibuya"},
    4:  { "city": "Mumbai", "name": "Goregaon"},
    5:  { "city": "Mumbai", "name": "Fort"}
}
####################################################################

class Employee:
    name : str 
    age : int
    ID : int
    city : int
    branches : list[int] # This is a list of branches (as branch codes) to which the employee may report
    salary : int 

    def __init__(self, name, age, ID, city,\
                 branchcodes, salary = None):
        self.name = name
        self.age = age 
        self.ID = ID
        self.city = city
        self.branches = branchcodes
        if salary is not None: self.salary = salary
        else: self.salary = 10_000 
    
    def change_city(self, new_city:str) -> bool:
        if self.city == new_city:
            return False
        self.city = new_city
        return True
        # Change the city 
        # Return true if city change, successful, return false if city same as old city
        pass

    def migrate_branch(self, new_code:int) -> bool:
        if len(self.branches) != 1:
            return False
        current_branch = self.branches[0]
        if branchmap[current_branch]["city"] == branchmap[new_code]["city"]:
            self.branches[0] = new_code
            return True
        return False
        # Should work only on those employees who have a single 
        # branch to report to. Fail for others.
        # Change old branch to new if it is in the same city, else return false.
        pass

    def increment(self, increment_amt: int) -> None:
        self.salary += increment_amt
        # Increment salary by amount specified.
        pass





class Engineer(Employee):
    position : str # Position in organization Hierarchy

    def __init__(self, name, age, ID, city,\
                 branchcodes, position= "Junior", salary = None):
        # Call the parent's constructor
        super().__init__(name, age, ID, city, branchcodes, salary)
        if position in ["Junior" , "Senior" , "Team Lead" , "Director"]:
            self.position = position
        else:
            print("Invalid Position")
        # Check if position is one of  "Junior", "Senior", "Team Lead", or "Director" 
        # Only then set the position. 

    
    def increment(self, amt:int) -> None:
        bonus_amt = int(amt*0.1)
        self.salary += amt + bonus_amt
        
        # While other functions are the same for and engineer,
        # and increment to an engineer's salary should add a 10% bonus on to "amt"
        pass
        
    def promote(self, position:str) -> bool:
        hierarchy = ["Junior" , "Senior" , "Team Lead" , "Director"]

        if position not in hierarchy:
            return False
        if hierarchy.index(position) <= hierarchy.index(self.position):
            return False
        self.position = position
        self.increment(int(0.3*self.salary))
        return True 
        # Return false for a demotion or an invalid promotion
        # Promotion can only be to a higher position and
        # it should call the increment function with 30% of the present salary
        # as "amt". Thereafter return True.
        pass



class Salesman(Employee):
    """ 
    This class is to be entirely designed by you.

    Add increment (this time only a 5% bonus) and a promotion function
    This time the positions are: Rep -> Manager -> Head.

    Add an argument in init which tracks who is the superior
    that the employee reports to. This argument should be the ID of the superior
    It should be None for a "Head" and so, the argument should be optional in init.
    """
    
    # An extra member variable!
    superior : int # EMPLOYEE ID of the superior this guy reports to

    def __init__(self, name, age, ID, city, branchcodes, salary = "None", position = "Rep", superior = "None"): # Complete all this! Add arguments

        super().__init__(name, age, ID, city, branchcodes, salary)

        if position in ["Rep" , "Manager" , "Head"]:
            self.position = position
        else:
            print ("Invalid Position")
        if superior is not None:
            self.superior = int(superior)
        else:
            self.superior = None
        pass
    
    def increment(self, amt:int) -> None:
        bonus_amt = int(amt*0.05)
        self.salary += amt + bonus_amt

        pass

    def promote(self, position:str) -> bool:
        hierarchy = ["Rep" , "Manager" , "Head"]

        if position not in hierarchy:
            return False
        if hierarchy.index(position) <= hierarchy.index(self.position):
            return False
        self.position = position
        self.increment(int(0.3*self.salary))
        return True 
        pass

    def find_superior(self) -> tuple[int, str]:
        if self.superior is None:
            return None, None
        for s in sales_roster:
            if s.ID == self.superior:
                return s.ID, s.name
        return None, None

    def add_superior(self) -> bool:
        if self.position == "Rep":
            target_position = "Manager"
        elif self.position == "Manager":
            target_position = "Head"
        else:
            return False  # Head has no superior

        for s in sales_roster:
            if s.position == target_position and s.city == self.city:
                self.superior = s.ID
                return True
        return False

    def migrate_branch(self, new_code: int) -> bool:
        if new_code not in self.branches:
            self.branches.append(new_code)
        return True

    





    
    
