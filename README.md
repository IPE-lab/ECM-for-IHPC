# **Digital twin-driven energy consumption management of integrated heat pipe cooling system for a data center**
## Running screenshots show
- **The structure of proposed PINN model**
- <img src="img/The structure of proposed PINN model.jpg" width="400" />
***
## Paper Support
- Original information: Digital twin-driven energy consumption management of integrated heat pipe cooling system for a data center
- Recruitment journal: Applied Energy
- Original DOI: https://doi.org/10.1016/j.apenergy.2024.123840
***
## Description of the project
The energy consumption management (ECM) for the integrated heat pipe cooling (IHPC) systems has become a significant cost-cutting strategy, given the growing demand for the decreased cooling and maintenance costs in data centers. However, the traditional ECM strategies lack an integration with the real-time information and the automatic feedback control, causing the risks of system operation difficult to diagnose and the potential for energy saving hard to exploit. In this respect, a digital twin approach was proposed to efficiently and automatically implement the ECM strategy for an IHPC system. First, a digital twin architecture was established to enable seamless integration and real-time interaction between the physical system and the digital twin. Secondly, the digital twin models of monitoring, simulation, energy evaluation and optimization were developed to drive the corresponding services. Finally, the approach was verified on an IHPC system operating in a real-life data center. It is found that the approach can automatically detect and justify the abnormal states of the IHPC system. Moreover, the approach can reduce the power consumption by 23.63 % while meeting the production requirements. The mean relative errors of the supply air temperature and the cooling capacity between the digital twin simulated and the on-site records are 1.43% and 1.46%, respectively. In summary, the proposed approach provides a digital twin workflow that can significantly improve the efficiency of the ECM strategy deployed on an IHPC system.
***
## Functions of the project
(A) Energy_calculation.py was used to fit the energy calculation formula to find the unknown coefficients in the energy calculation formula. Where the Levenberg-Marquardt algorithm was used to solve the fitting problem.
Function Description:
1 # Set the form of the formula to be fitted. 
2 # Import the historical energy consumption data and preprocess it.
3 # To start the fitting work, the available libraries are: statsmodels, sklearn and scipy.

(B) GA_IDC.py was used to optimize the energy consumption of the cooling system. Specifically, the optimization algorithm uses the genetic algorithm. The objective function includes the calculated values of the simulation model obtained from the K-spice software.
Function Description:
1 # Send a GET request to connect to the k-spice and get the values (T_s and T_r) required in the objective function.
2 # Import the historical energy consumption data and preprocess it.  Define the constraints and penalty functions and generate the fitness function.
3 # Initialize the population and set up the selection, crossover and mutation operations.
4 # Setting up the adaptive change rules for the coefficients of the penalty function.
5 # Setting the relevant control parameters of the genetic algorithm for optimization work. 
***
## Environments
-	Python == 3.8.0
- pandas == 1.2.4
-	numpy == 1.24.4
-	matplotlib == 3.5.1
- scipy== 1.5.2
- requests==2.25.1
- statsmodels==0.12.0
- scikit-learn==0.23.2
***
## Use
- 1. Read the energy consumption history of the cooling system
- 2. Getting the unknown coefficients in the energy consumption equation
- 3. Connecting the simulation model built in k-spice software
- 4. Call GA_IDC.py to optimize the energy consumption of the cooling system
