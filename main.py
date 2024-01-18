import customtkinter
from CTkListbox import *
import warnings
from tkinter import messagebox, ttk
import math
from pymongo import MongoClient, errors
import hashlib
import backend

# Ignore all warnings
warnings.filterwarnings("ignore")

customtkinter.set_default_color_theme("green")
customtkinter.set_appearance_mode("light")


class App(customtkinter.CTk):
    APP_NAME = "Job Recommendation"
    WIDTH = 800
    HEIGHT = 600

    def __init__(self, *args, **kwargs):
        """
        Initializes the App object.

        Args:
            *args: Variable length argument list.
            **kwargs: Arbitrary keyword arguments.
        """
        super().__init__(*args, **kwargs)

        # Set the title, dimensions, and icon for the application window
        self.title(App.APP_NAME)
        # Set the window size to the screen size
        screen_width = self.winfo_screenwidth() - 10
        screen_height = self.winfo_screenheight() - 70
        self.geometry(f"{screen_width}x{screen_height}+0+0")
        self.minsize(App.WIDTH, App.HEIGHT)
        # self.iconbitmap("img/icon.ico")

        # Configure behavior on window close
        self.protocol("WM_DELETE_WINDOW", self.on_closing)
        self.createcommand('tk::mac::Quit', self.on_closing)

        # Configure grid weights for row and column resizing
        self.grid_rowconfigure(0, weight=0)
        self.grid_rowconfigure(1, weight=1)
        self.grid_columnconfigure(0, weight=1)
        self.grid_columnconfigure(1, weight=2)
        self.grid_columnconfigure(2, weight=1)

        # ================ LEFT ================
        # Create and configure the left frame for user-related actions
        self.left_frame = customtkinter.CTkFrame(self, fg_color="#BDBDBD")
        self.left_frame.grid(row=0, column=0, rowspan=2, padx=0, pady=0, sticky="nsew")

        # Initialize variables related to user skills and login status
        self.skills_entry = None
        self.skills_listbox = None
        self.skills = []
        self.search_history = []
        self.username_entry = None
        self.password_entry = None
        self.connected = False

        # ================ TOP ================
        # Create and configure the top frame for search and options
        top_frame = customtkinter.CTkFrame(self)
        top_frame.grid(row=0, column=1, padx=0, pady=0, sticky="nsew")

        # Create and configure search input and button
        search_frame = customtkinter.CTkFrame(top_frame, fg_color='transparent')
        search_frame.pack(side="top", padx=10, pady=5)
        self.search_entry = customtkinter.CTkEntry(search_frame, width=350, placeholder_text='Ex: java developer')
        self.search_entry.grid(row=0, column=0)
        self.search_entry.bind("<Return>", self.search)
        search_button = customtkinter.CTkButton(search_frame, text="Search", command=self.search, width=75)
        search_button.grid(row=0, column=1)

        # Create and configure option menus for location and company
        option_frame = customtkinter.CTkFrame(top_frame, fg_color='transparent')
        option_frame.pack(side="top", padx=10, pady=5)

        company_label = customtkinter.CTkLabel(option_frame, text="Company")
        company_label.grid(row=0, column=0, padx=5)
        self.company_combobox = customtkinter.CTkComboBox(option_frame,
                                                          values=["All"] + sorted(backend.get_company().tolist()),
                                                          state="readonly",
                                                          button_color="#30cc84")
        self.company_combobox.grid(row=0, column=1, padx=10)
        self.company_combobox.set("All")

        location_label = customtkinter.CTkLabel(option_frame, text="Location")
        location_label.grid(row=0, column=2, padx=5)
        self.location_combobox = customtkinter.CTkComboBox(option_frame,
                                                           values=["All"] + sorted(backend.get_location().tolist()),
                                                           state="readonly",
                                                           button_color="#30cc84")
        self.location_combobox.grid(row=0, column=3, padx=10)
        self.location_combobox.set("All")

        # ================ CONTENT ================
        # Create and configure the content frame for displaying job offers
        self.content_frame = customtkinter.CTkFrame(self)
        self.content_frame.grid(row=1, column=1, padx=0, pady=0, sticky="nsew")

        # Initialize variables related to content display
        self.choose_page = None
        self.main_frame = None
        self.job_offers = None

        # Set the number of results per page and display initial content
        self.result_by_page = 10

        self.display_content(backend.search(input_text=None))

        # ================ RIGHT ================
        # Create and configure the right frame for displaying skill-based job offers
        self.right_frame = customtkinter.CTkFrame(self, fg_color="#BDBDBD")
        self.right_frame.grid(row=0, column=2, rowspan=2, padx=0, pady=0, sticky="nsew")
        self.right_skill_frame = customtkinter.CTkFrame(self.right_frame, fg_color='transparent')
        self.right_skill_frame.pack(fill="x", side="top")

        # Display skill-based job offers on the right side
        self.display_right(self.get_skills_job())

        # Create and configure the right frame for displaying search-recommend job offers
        self.right_search_frame = customtkinter.CTkFrame(self.right_frame, fg_color='transparent')
        self.right_search_frame.pack(fill="x", side="bottom")


        # Display the account menu
        self.account_menu()

    def add_research(self, search):
        """
        Adds a new search to the search history, updates the display, and, if connected to a database, updates the database.

        Parameters:
        - search (str): The search query to be added to the history.
        """
        # Update the search history
        if len(self.search_history) == 5:
            self.search_history = [search] + self.search_history[:-1]
        else:
            self.search_history = [search] + self.search_history

        # Update the display with the latest search job
        self.display_right_search(self.get_search_job())

        # If the instance is connected to a database
        if self.connected:
            # Establish a connection to the database
            collection = connexion_bdd()

            # If unable to connect to the database, exit the function
            if collection is None:
                return

            # Update the search history in the database for the connected user
            collection.update_one({'name': self.connected}, {'$set': {'search_history': self.search_history}})

    def get_search_job(self):
        """
        Retrieves a list of job offers based on user search history.

        Returns:
            list: A list of job offers.
        """
        if not self.search_history:
            return [{'jobtitle_old': 'Use search bar', 'company': ''}]
        else:
            job_offers_list = backend.recommendation_search(self.search_history)
            return job_offers_list

    def get_skills_job(self):
        """
        Retrieves a list of job offers based on user skills.

        Returns:
            list: A list of job offers.
        """
        if not self.skills:
            return [{'jobtitle_old': 'Insert skills', 'company': 'Go to \'create profile\''}]
        else:
            job_offers_list = backend.recommendation_skills(self.skills)
            return job_offers_list

    def search(self, event=None):
        """
        Initiates a search for job offers based on user input.

        Args:
            event: The event triggering the search (if bind) (default is None).
        """
        search = self.search_entry.get()
        self.search_entry.delete(0, "end")
        company = self.company_combobox.get()
        self.company_combobox.set("All")
        location = self.location_combobox.get()
        self.location_combobox.set("All")
        company = (None if company == "All" else company)
        location = (None if location == "All" else location)
        result = backend.search(search, company=company, joblocation_address=location)
        if result is None:
            result = [{'jobtitle_old': 'No result', 'company': '', 'jobdescription_old': ''}]
        else:
            self.add_research(search)
        self.display_content(result)


    def display_right(self, job_offers):
        """
        Displays job offers regarding skills on the right side of the application.

        Args:
            job_offers (list): A list of job offers to display.
        """
        # Clear existing content in the right frame
        for i in self.right_skill_frame.winfo_children():
            i.destroy()

        # Set the maximum number of job offers to display
        number_display = 5

        # Create and display a title label
        title_label = customtkinter.CTkLabel(self.right_skill_frame, text="Suggestions based on your skills",
                                             font=("Arial", 10, "bold"))
        title_label.pack()

        # Iterate through job offers and display them
        for i, job_offer in enumerate(job_offers):
            if i + 1 <= number_display:
                # Create a frame for each job offer
                job_frame = customtkinter.CTkFrame(self.right_skill_frame, fg_color='transparent')
                job_frame.pack(fill="x", side="top", padx=10)

                # Display job title with bold font
                title_label = customtkinter.CTkLabel(job_frame, text=job_offer["jobtitle_old"], font=("Arial", 10, "bold"))
                title_label.grid(row=0, column=0, columnspan=2, sticky="w")

                # Display company information
                company_label = customtkinter.CTkLabel(job_frame, text=job_offer["company"],
                                                       font=("Arial", 10))
                company_label.grid(row=1, column=0, columnspan=2, sticky="w")

    def display_right_search(self, job_offers):
        """
        Displays job offers regarding search history on the right side of the application.

        Args:
            job_offers (list): A list of job offers to display.
        """
        # Clear existing content in the right frame
        for i in self.right_search_frame.winfo_children():
            i.destroy()

        # Set the maximum number of job offers to display
        number_display = 5

        # Create and display a title label
        title_label = customtkinter.CTkLabel(self.right_search_frame, text="Suggestions based on your research",
                                             font=("Arial", 10, "bold"))
        title_label.pack()

        # Iterate through job offers and display them
        for i, job_offer in enumerate(job_offers):
            if i + 1 <= number_display:
                # Create a frame for each job offer
                job_frame = customtkinter.CTkFrame(self.right_search_frame, fg_color='transparent')
                job_frame.pack(fill="x", side="top", padx=10)

                # Display job title with bold font
                title_label = customtkinter.CTkLabel(job_frame, text=job_offer["jobtitle_old"], font=("Arial", 10, "bold"))
                title_label.grid(row=0, column=0, columnspan=2, sticky="w")

                # Display company information
                company_label = customtkinter.CTkLabel(job_frame, text=job_offer["company"],
                                                       font=("Arial", 10))
                company_label.grid(row=1, column=0, columnspan=2, sticky="w")

    def update_content(self, page="1"):
        """
        Updates the content display based on the selected page.

        Args:
            page (str): The selected page (default is "1").
        """
        # Clear existing content in the main frame
        for i in self.main_frame.winfo_children():
            i.destroy()

        # Convert page to integer
        page = int(page)

        # Calculate the range of job offers to display on the current page
        page_min = 1 + (page - 1) * self.result_by_page
        page_max = self.result_by_page + (page - 1) * self.result_by_page

        # Iterate through job offers and display them based on the selected page
        for i, job_offer in enumerate(self.job_offers):
            if page_min <= i + 1 <= page_max:
                # Create a frame for each job offer
                job_frame = customtkinter.CTkFrame(self.main_frame)
                job_frame.pack(fill="x", side="top")

                # Display job title with bold font
                title_label = customtkinter.CTkLabel(job_frame, text=job_offer["jobtitle_old"], font=("Arial", 14, "bold"))
                title_label.grid(row=0, column=0, columnspan=2, sticky="w")

                # Display company information
                company_label = customtkinter.CTkLabel(job_frame, text=job_offer["company"])
                company_label.grid(row=1, column=0, columnspan=2, sticky="w")

                # Display job description with word wrap
                description_label = customtkinter.CTkLabel(job_frame, text=job_offer["jobdescription_old"], wraplength=500,
                                                           justify="left")
                description_label.grid(row=2, column=0, columnspan=2, sticky="w")
                separator = customtkinter.CTkLabel(job_frame, text="--------------------"*100)
                separator.grid(row=3, column=0, columnspan=2, sticky="w")


    def display_content(self, job_offers):
        """
        Displays job offers in the main content area.

        Args:
            job_offers (list): A list of job offers to display.
        """
        # Clear existing content in the content frame
        for i in self.content_frame.winfo_children():
            i.destroy()

        # Store the received job offers in the instance variable
        self.job_offers = job_offers[:50]

        # Calculate the number of pages based on the result_by_page setting
        number_by_page = math.ceil(len(self.job_offers) / self.result_by_page)

        # Create a list of page values for the segmented button
        values = [str(i + 1) for i in range(number_by_page)]

        if len(values) > 1:
            # Create and display the segmented button for page selection
            self.choose_page = customtkinter.CTkSegmentedButton(self.content_frame, values=values,
                                                                command=self.update_content)
            self.choose_page.pack(fill='x')
            self.choose_page.set("1")

        # Create a scrollable frame to display the job offers
        self.main_frame = customtkinter.CTkScrollableFrame(self.content_frame, fg_color='transparent')
        self.main_frame.pack(fill='both', expand=True)

        # Update content to display job offers on the initial page
        self.update_content()

    def login_profile(self):
        """
        Displays the login interface for the user.
        """
        # Clear existing content in the left frame
        for i in self.left_frame.winfo_children():
            i.destroy()

        # Create and display labels and entry widgets for username and password
        username_label = customtkinter.CTkLabel(self.left_frame, text="Username:")
        username_label.pack()
        self.username_entry = customtkinter.CTkEntry(self.left_frame, width=150)
        self.username_entry.pack()

        password_label = customtkinter.CTkLabel(self.left_frame, text="Password:")
        password_label.pack()
        self.password_entry = customtkinter.CTkEntry(self.left_frame, width=150, show="*")
        self.password_entry.pack()

        # Bind the "Return" key to the login function
        self.password_entry.bind("<Return>", self.login)

        # Create and display login button
        create_button_insert = customtkinter.CTkButton(self.left_frame, text="Login", command=self.login)
        create_button_insert.pack(pady=10)

        # Create and display back button, linking to the account menu
        back_button = customtkinter.CTkButton(self.left_frame, text="Back", command=self.account_menu)
        back_button.pack(pady=10)

    def login(self, event=None):
        """
        Handles the user login process.

        Args:
            event: The event triggering (if bind) the login (default is None).
        """
        # Retrieve the entered username and password
        username = self.username_entry.get()
        password = self.password_entry.get()

        # Establish a connection to the database
        collection = connexion_bdd()

        # If unable to connect to the database, exit the function
        if collection is None:
            return

        # Verification of the entered username and password
        if not collection.find_one({"name": username}):
            messagebox.showwarning("Attention", "The entered username does not exist.")
        elif collection.find_one({"name": username, "pass": hashlib.sha224(password.encode('utf-8')).hexdigest()}):
            # If the username and password match, retrieve user skills and display the profile
            self.skills = collection.find_one({"name": username})['skills']
            self.search_history = collection.find_one({"name": username})['search_history']
            self.connected = username
            self.display_profile()
            self.display_right(self.get_skills_job())
            self.display_right_search(self.get_search_job())
        else:
            # Display a warning if the entered password does not match the username
            messagebox.showwarning("Attention", "The entered password does not match the username " + username)

    def create_profile(self):
        """
        Displays the interface for creating a user profile.
        """
        # Clear existing content in the left frame
        for i in self.left_frame.winfo_children():
            i.destroy()

        # Create and display labels and entry widgets for username and password
        username_label = customtkinter.CTkLabel(self.left_frame, text="Username:")
        username_label.pack()
        self.username_entry = customtkinter.CTkEntry(self.left_frame, width=150)
        self.username_entry.pack()

        password_label = customtkinter.CTkLabel(self.left_frame, text="Password:")
        password_label.pack()
        self.password_entry = customtkinter.CTkEntry(self.left_frame, width=150, show="*")
        self.password_entry.pack()

        # Bind the "Return" key to the register function
        self.password_entry.bind("<Return>", self.register)

        # Create and display create button
        create_button_insert = customtkinter.CTkButton(self.left_frame, text="Create", command=self.register)
        create_button_insert.pack(pady=10)

        # Create and display back button, linking to the account menu
        back_button = customtkinter.CTkButton(self.left_frame, text="Back", command=self.account_menu)
        back_button.pack(pady=10)

    def register(self, event=None):
        """
        Handles the user registration process.

        Args:
            event: The event triggering (if bind) the registration (default is None).
        """
        # Retrieve the entered username and password
        username = self.username_entry.get()
        password = self.password_entry.get()

        # Check for empty username or password fields
        if username.replace(" ", "") == "":
            messagebox.showwarning("Attention", "You must fill in the 'Username' field!")
        elif password.replace(" ", "") == "":
            messagebox.showwarning("Attention", "You must fill in the 'Password' field!")

        # Check the length of the username
        elif len(username) > 15:
            messagebox.showwarning("Attention", "You must enter a username with less than 15 characters.")

        else:  # If no issues
            # Connect to the database
            collection = connexion_bdd()

            # If unable to connect to the database, exit the function
            if collection is None:
                return

            # Check if the username already exists
            if collection.find_one({"name": username}):
                messagebox.showinfo("Attention", "Username already in use!")
            else:
                # Insert the new account with default parameters
                collection.insert_one({"name": username,
                                       "pass": hashlib.sha224(password.encode('utf-8')).hexdigest(),
                                       "skills": [],
                                       "search_history": []
                                       })

                # Log in the newly registered user
                self.login()

                # Display a success message
                messagebox.showinfo("New Account", "Your account has been successfully registered.")

    def display_profile(self):
        """
        Displays the user's profile information and skills.
        """
        # Clear existing widgets in the left frame
        for i in self.left_frame.winfo_children():
            i.destroy()

        # Display user's skills entry and insert button
        skills_label = customtkinter.CTkLabel(self.left_frame, text="Skills:")
        skills_label.pack()
        self.skills_entry = customtkinter.CTkEntry(self.left_frame, width=150)
        self.skills_entry.pack()
        self.skills_entry.bind("<Return>", self.insert_skill)

        skills_button_insert = customtkinter.CTkButton(self.left_frame, text="Insert", command=self.insert_skill)
        skills_button_insert.pack(pady=10)

        # Display user's skills listbox
        self.skills_listbox = CTkListbox(self.left_frame, width=150, text_color="#000")
        self.skills_listbox.pack()
        for i in range(len(self.skills)):
            self.skills_listbox.insert(i, self.skills[i])

        # if from main to guest, reset research history
        if not self.connected:
            self.search_history = []
            self.display_right_search(self.get_search_job())

        # Display remove button for user's skills
        skills_button_remove = customtkinter.CTkButton(self.left_frame, text="Remove", command=self.remove_skill)
        skills_button_remove.pack(pady=10)

        # Display back button to return to the account menu
        back_button = customtkinter.CTkButton(self.left_frame, text="Back", command=self.account_menu)
        back_button.pack(pady=10)

        # Display delete account button if user is connected
        if self.connected:
            delete_button = customtkinter.CTkButton(self.left_frame, text="Delete account", command=self.delete_account)
            delete_button.pack(pady=10, side="bottom")

    def delete_account(self):
        """
        Deletes the user's account.
        """
        # Display confirmation dialog
        if messagebox.askyesno('Attention', f'Are you sure you want to delete the account: {self.connected}',
                               default='no'):
            # Attempt to connect to the database
            collection = connexion_bdd()
            if collection is None:
                return

            # Delete the user's account from the database
            collection.delete_one({"name": self.connected})

            # Return to the account menu
            self.account_menu()

            # Display success message
            messagebox.showinfo('Information', 'Your account has been successfully deleted.')

    def remove_skill(self):
        """
        Removes a skill from the user's profile.
        """
        # Check if a skill is selected in the skills listbox
        if self.skills_listbox.curselection() is not None:
            # Remove the selected skill from the skills list
            self.skills.pop(self.skills_listbox.curselection())

            # Delete the selected skill from the skills listbox
            self.skills_listbox.delete(self.skills_listbox.curselection())

            # Update the display on the right side based on the modified skills list
            self.display_right(self.get_skills_job())

            # If the user is connected, update the skills in the database
            if self.connected:
                collection = connexion_bdd()
                if collection is None:
                    return
                collection.update_one({'name': self.connected}, {'$set': {'skills': self.skills}})
        else:
            # Display a warning if no skill is selected
            messagebox.showwarning("Not selected", "You need to select an element !")

    def insert_skill(self, event=None):
        """
        Inserts a skill into the user's profile.

        Args:
            event: The event triggering (if bind) the skill insertion (default is None).
        """
        # Retrieve the skill from the input field and remove leading/trailing spaces
        skill = self.skills_entry.get().replace(" ", "")

        # Check if the skill is not empty
        if skill != "":
            # Check if the skill is not already in the user's skills list
            if skill not in self.skills:
                # Insert the skill into the skills listbox and add it to the skills list
                self.skills_listbox.insert(len(self.skills), skill)
                self.skills.append(skill)

                # Update the display on the right side based on the modified skills list
                self.display_right(self.get_skills_job())

                # Clear the input field
                self.skills_entry.delete(0, 'end')

                # If the user is connected, update the skills in the database
                if self.connected:
                    collection = connexion_bdd()
                    if collection is None:
                        return
                    collection.update_one({'name': self.connected}, {'$set': {'skills': self.skills}})
                    collection.update_one({'name': self.connected}, {'$set': {'search_history': self.search_history}})
            else:
                # Display a warning if the skill is already in the user's skills list
                messagebox.showwarning("Warning", "You already have this skill!")
        else:
            # Clear the input field if the skill is empty
            self.skills_entry.delete(0, 'end')

    def account_menu(self):
        """
        Displays the account menu, allowing the user to navigate between actions.
        """
        self.search_history = []
        self.display_right_search(self.get_search_job())
        if not self.skills:
            # If the user has no skills, display options to create a profile, login, or continue as a guest
            for i in self.left_frame.winfo_children():
                i.destroy()

            create_button = customtkinter.CTkButton(self.left_frame, text="Create profile", command=self.create_profile)
            create_button.pack(pady=10)

            login_button = customtkinter.CTkButton(self.left_frame, text="Login", command=self.login_profile)
            login_button.pack(pady=10)

            guest_button = customtkinter.CTkButton(self.left_frame, text="Guest", command=self.display_profile,
                                                   fg_color="#6495ED", hover_color="#004d99")
            guest_button.pack(pady=10)
        else:
            if not self.connected:
                # If the user has skills but is not connected, prompt a warning about skills deletion
                if messagebox.askokcancel("Warning", "If you go back, your skills will be deleted"):
                    self.skills = []
                    self.connected = False
                    self.display_right(self.get_skills_job())
            else:
                # If the user is connected, clear skills, disconnect the user, and update the display
                self.skills = []
                self.connected = False
                self.display_right(self.get_skills_job())

            # Recursively call the account_menu to refresh the display after performing actions
            self.account_menu()

    def on_closing(self, event=0):
        """
        Handles the closing event of the application.

        Args:
            event: The event triggering the closing (default is 0).
        """
        self.destroy()

    def start(self):
        """
        Starts the application's main loop.
        """
        self.mainloop()


def connexion_bdd(showmessageerror=True):
    """
    Enables the connection with the MongoDB database.

    Args:
        showmessageerror (bool): Authorization to display an error message or not.

    Returns:
        pymongo.collection.Collection: The collection from the database.
    """
    try:
        # MongoDB connection parameters
        connection_string = "mongodb+srv://admin:tuV40I2VAd6Ui93i@ipsa.2qm9k.mongodb.net/myFirstDatabase?retryWrites=true"
        connect_timeout = 5000  # Connection timeout in milliseconds

        # Establishing connection to the MongoDB cluster
        cluster = MongoClient(connection_string, connectTimeoutMS=connect_timeout)

        # Retrieving the database and collection from the cluster
        database = cluster['JobRecommendation']
        collection = database['users']

        return collection

    except errors.ConfigurationError:
        # If unable to connect, display an error message (if authorized) and return None
        if showmessageerror:
            messagebox.showerror("Connection Error", "Unable to connect to the database.\n\n"
                                                     "Please check your internet connection.")
        return None


if __name__ == "__main__":
    app = App()
    app.start()
