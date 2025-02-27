from setuptools import find_packages, setup

"""
░▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓██▒▒  ░░▒░░░▒░▒▒▒▓███▓▒▒▒▒▒▒▒▒░▒ ░█▓▒▒▒▓▓░            ░░░░░░░░░░░░░     ░░░░      ░ ░░░░                  
░▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓██▒░░▒▓▓▓▓▓▒▒▒▒▒▒▒▒▒▒▒▓▓█▒▒▒▒▒▒▒▒░ ▒▓▒▒▒▒▓▓░ ░░░        ░░░░░░░░░░░░ ░░░ ░░ ░░   ░░░░░   ░░▓▓▓▓▓▒          
░▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓█▒░ ░▒▓▓▓▓▓▓▓▓▓▓▒▒▒▓▓▓▓▓▓▓▓▓▓▓▒▒▒▒ ░███▓▓██▓░░░░         ░░░ ░░░░ ░░░░░░░░░░░░   ░░░   ░▒▓▓▓▓▓▓▓█▓░        
░▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓█░░          ░▓▓▓▓▓▓▓▓▒▒▓▓▓███▓▓▒▒▒░▓███▓▓▓▓▒▒▒░         ░▒▒░░░░▒░░░░░░░░░░░░░░  ░░  ▒▓▓▓▒▒▒▒▒▒░▒▓▓░       
░▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓█░░          ▒▓█▓▒░░▒▓▓▓▓▓▒▒▒░ ░░ ░       ░░░░░░  ░░░ ░░  ▒▒░ ░ ░░░░░░░▒░░  ░░▒    ░▓▓▓▓▒▒▒▒▒▒▒▒▒▒▓▓       
░▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓█▒         ░         ░░░░░░░                                                     ▒▓▓▓▒▒▒▒▒▒▒▒▒▒▒▒▒▒▓▒      
░▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓█▒              ▒▒▓█▓▓▓▓▒                                                      ░▓▓▓▒▒▒░░▒░░▒▒▒▒▒▒▒▒▓▓      
░▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓█▒░           ░▒▓▓▓▓▒░░                   ░▒▓░▓▓▒░▓▒▒▒▓▓▓▒░          ░        ▓▓▒▒░░░░░░░░▒▒▒▒▒▒▓▒▒▒▓      
░▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓█▓░                                     ░██▓▓▓▓▓▓▓▓▓▓▓█▒░░░          ░      ▓▓▓▒░░░░▒▒▒▒▒▒▒▒▓▒▓▒▒░▒▒▓      
▒▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▒                                 ▒▓▒▓▓▓▓░▒▒▒▒▒░▒▒▒▒▒              ░    ▒▓▓▒░░▒▒▒▒▒▒▒▒▓▓▓▓▓▓▒▒░░░▓▓ ░░░░ 
░▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▒▓▓▓▒░▒                          ▒█░▒▒▒▓▓▓▓▒░▒░░░░░░░░░░░░                   ▓▓▒░░▒▒▒▒▒▓▓▓▓▓█▒▒░░░░░░░▒▓▒      
░▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▒▒▒▒░▒░                       ░▓█▓▒▓▓▓▒▒▒▒░░░   ░░░░▒▒▒░                   ░▒░░▒░▒▒▒▒▒░░░░░░░░░░░░▒░░▒▒       
▒▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▒▒▒▒▒▓█▒▒░░░▒                      ░▓██▓░░▒▓▓▒░░░░░    ░░░░░▒▒░░░░░              ░▒░░░░░▒░░░░░░  ░░   ░░░ ░░▓    ▓██ 
▒▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▒▒▒░▒▓▒░░▒░                    ░▒███▓▒▒▒▓▓▒▒ ░▒░     ░░▒░░▒░▒▒▒░░░▒░         ▒▒▒░░░░▒▒░▒▒▒▒▒▒▓▓▓▓▓▓▓▓▒▒▒▓██████▒   
▒▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▒▒▓▓▓▓▓▓▒░░░░░▒░░     ░░           ▒▓██▓▒▒▒▒▓▓▓▒▒░▒▓      ░░░░▒▒░░▒▒░░▒░         ░░   ░░░░▒▒▒▒▒▒▒▒▒▒▒▒▒░▒▒▒▒▒▒▒░        
▒▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▒▒▒▒▒▓▒▒░░▒▒░      ░▒░     ░  ▒████▓▓▓▒░ ░▒▒▒▒▒▒░      ░░░░░░▒▒▒░░░░░░░░    ░   ░   ░░░░░▒▒▒▒▒▒▓▓▓▓██▓▓▓▒▒▒███▓▓▓▓▓▒ 
▒▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▒░░░▓░░▒▒░░        █▓░    █████████▓▓▓▒  ░   ░          ░   ░  ░ ░░░░░ ░░░░░░▒░░░░▒▒▒▒░░░░░░░▒▓▓▓▒▓████████████▓█▓▓ 
▒▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▒▒▒▓▓▒░░░▒░          ▒██▓▒  ▓███████▓█▓▓▓▒░▒░   ░                  ░░░  ░░░░░░▒░░░░▒░░░░░░░░░░░░░   ░▒▒▒▒▒▓▓██████████ 
▒▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▒▒▓▓▓▒▒▒▒       ▒   ██▓▒░▒███████▓▓▓▓▒▒▒▒░                            ░     ░░░░░░░░░░░░░░░▒░▒▒▒▒░░░░░░░░░░▒         
▒▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▒▒░░░░▒░  ░░▒   ░ ▒░  █████████████▓▓▒▒▒▒▓▒                                    ░░░░▒▒░░▒▒▒▒▒▒▒▒░░░░░░▒▒▒▒▓▒▒░▒         
▒▓▓▓▓▓▓▓▓▓▓▓▓▓▓▒▒░░░░░░░       ░▓▓  ▒▓▓▓██████████████▒▒▒▒▒▓██░                                   ░░░   ░░▒░░▓▓▓▓▓▓▓▓▒░░   ░▒▒▒▒▓▓░░░░    
▒▓▓▓▓▓▓▓▓▓▓▓▓▒▓▓▒▒▒▓▓▓██▒▓▓░     ▓█▒▒▓▒▒▒█████████▓▓▓▒░▒▒▒██▓░                                        ░░░     ░  ░░░▒▓█▓▒▒▓▓▒▒▒░▒░░       
▒▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓█▓▒░░▒▒░ ░▓▒    ▓██▓▒▒█████████████▓░▒▓▓▓▓░                                                           ░░░░░░░░░░░░      
▒▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▒▒░▒░░░░     ░▒   ▓██▓▓████████████████▓▓██▒                                                       ░░░░░▒▒▒▒░░▒▒▒░░      
▒▓▓▓▓▓▓▓▓▓▒▓▓▒▓▒▒▒▒░░░▒░           ▒█▓▓▓██████████████▓▓▓██░                                                  ░░░       ░░▒▒▒▓▒▓███▒      
▒▓▓▓▓▓▓▓▓▓▓▓▓▒▓▓▓▓▓▒▓▒▒░           ▒▓▓█▓███████████▓▓▓▓▓▓██▒░                                                               ░▒▒▓▒▒░       
▒▓▓▓▓▓▓▓▓▓▓▓▓▓▓▒▒░░░▒░         ░▒▓▒▒▓▒▒▓████████████▓▓▒▓▓█▓░                                                           ░░░▒▒▒░░░▒▓▓░      
▒▓▓▓▓▓▓▓▓▓▓▓▓▓▓▒▒▒▒▒░            ▓██▓▒▒▒██████████▓▓▓▒▓▓██▒                                                            ▒▓██████████▒      
▒▓▓▓▓▓▓▓▓▓█▓▓▓▓▒░░▒            ░▒▓█████▒▓█████████▓▓▓▒▒██▒                                                            ░▒▒▓█▓▓████████▒    
▒▓▓▓▓▓▓▓▓▓▓▓▓▓▓██▒░                  ░█▓▓██████████▓▓▓▒█▒                ██░                                             ░▒▓▓█▓█▓█████░   
▒▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▒░          ░▓▒       ▒▓██████████▓▓███░         ▒▓▓▓█░                                                  ░░░▒▓▓██████░   
▒▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▒▒             ░░░░▒▒░░▒█▓██████████████▒        ▒▓▒░                                                       ░▒▓▓▓▓▓█████▒ 
▒▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▒▒                     ▒█████████████████░                                                                      ░▒▓████▓▒ 
▒▓▓▓▓▓▓▓▓▓▓▓▓▓▓█▒▒      ░▒░▒░   ░      ▓█████████████████                                                                           ░▒██▓ 
▒▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▒▒       ░▒███████████▓██▓▓██████████████         ░░░                                                            ░▒░      
▒▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▒▒   ░░░░▓███████████▓▓███████████████▓█▓    ▒▒▒▒░▒░                                                                ▒░░░▒ 
▒▓▓▓▓▓▓▓▓▓▓▓▓▓▓█▒▒     ▒▓███████████▓▓██████████████▓██▒▓████████▒░ ░░░                                                                   
▒▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▒▒   ░░▓████████████▓▓███████████████████████████▓▒                                                                       
▒▓▓▓▓▓▓▓▓▓▓▓▓▓▓▒░░   ░█████████████▓▓███████████████████████████████▓▒░░                                                                  
▒▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▒▒░░▓██████████████▓████████████████████████████████▓▒▒░▒▒▒░                                                              
▒▓▓▓▓▓▓▓▓▓▓▓▓▓▓██▓▒░▓██████████████▒▓█▓█▓▓▓▓█████████████████████████▓▓▒                                                                  
▒▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓███▒░▒▓███████████▒░▒▓▒▒▒▓▓▓▓▓▓▓▓▓▓███████████████████▓▓▓▒░                                                               
▒▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓█▒░▒▒▒▓████▓████▓░░▒▒▒▓▓█▓▓▒░  ▒█████████████████████████▓▒ ░    ░  ░  ░                                                
▒▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓██▓▒▒▒▒▓▓▓██▓▓███▓░░░▒░▒░     ░▓▓▓▓▓▓▒▒▓██████▓██▓▓▓██▓▓▓███▓▓░                                     ░                   
▒▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▒▒░░░░░ ░▓██████▓░░░░░░  ░░▒▒▒▒▒▒▒▒▒▓▓█████████████████████████████████████████▓▒▒▓▓▓▓▒▒▓▓▓▓▓▒░░                      
▒▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓█▓▒▒░░▒░▓████▓▓▓███░  ░░░░▓▓▓▒▒▒▒░░░░░▒▒▒▒▓▓▓▓██████▓█▓███▓██▓▓▓▓▓▒░                                                   
▒▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▒▒▓▓▓▓▒▒▒▒████████▓░ ░░▓▓▒▒▒▒▒▒▒▒▒▓▓███████████▓▓█▓███▓▓▓▓▓▓▓▓█████▓▒░░                                              
▒▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▒▒▒▒▓█▓██▓▓▓▓▓██▓░░░▒▓▒▒▒▓▓▓███▓▓▓▓▓▓▓▓██████████████▓▓▓▓▓██▓▓▓▓████▓▒░                                           
▒▓▓▓▓▓▓▓▓▓▓▓▓▓▓█▓▓▓▓██▓▓▒▒▒▒▒▒░▒░▒▒▒▓▒▓▓▒▒░▒▒▒▒▓▓▓▓▓▓██████▓███▓▓▓▓▓▓▓▓▓███████▓▓▓▓▓▓██▓▓▓▓▓▓█████▒░                                      
▒▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▒▒▒▒▒▒▒▓▓▓▓▓▓▓▓▓▒▒░ ░░▒▒▒▓▓██▓▓█▓▓▓▓▓▓████████████▓▒▓▒▓█████▓▓▓██▓▓█▓▓▓▓▓▓▓███████████▓▒░░░░░                ░░░▓ 
▒▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓█▓▓▓▓▒░▒▒▒░░░░░▒▒▒▒▒░░  ░░░░░▒░▒▒▓▓▓████▓██████████▓▓▓▒▒▒▓▓▓▓▓▓██▓█▓▓▓▓████▓▓▓▓▓▓▓▓▓▓██▓███████▓▓▒▒░      ░▒░    ░▒▒▒ 
▒▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▒▒▒▒▒▒▒▒▒▒▒▒░░░░░░░▒░░░░░░░░░░░░▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒██▓█████▓▓▓█████▓▓▓██▓████▓▓▓▓▓▓▓▓██████▓▓██▓█▓▓░░░  ░▒███▓▓▒▒▓▒
▒▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▒▒▓▒▒▒▒▒▒░░▒▒▒░▒░░░░░░░░▒░░░░░░░░░▒▒▒▒▒▒▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓█▓▓█████████▓██████▓▓▓██████████▓██▓▓██▓▓▒▒▒▒░▒▒██████▒
▒▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▒▒▓▓▓▓▒▒▒▒░▒▒▒▒▒░░▒▒▒▒▒▒▒▒▒▒▒▒▓▓▒▒▒▒▒▓▓▓▓▓▓▓▓▓█▓▓▓▓▓▓▓▓▓▓█▓▓▓▓▓███████▓▓▓█████████▓██████████▓████▓▓▓█▓▓▓▓████▓
▒▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▒░▒▒▒▒▒▒▒▒▒░▒▒▒▒▒▒▒▒▒▒▓▓▓▓▓▓▓█████▓█▓▓▓▓▓▓▓▓▓▓▓████████▓▓█▓▓███▓▓▓██████▓▓█▓██▓▓▓████████▓▓▓▓██▓▓████▓
▒▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▒▓▓▒▒▓▓▓▓▒▒▒▒▒▒▒▒▒▒▓▒▒▒▒▒▓▒▒▒▒▒▒▓▒▓▓▓▓▓▓▒▓▓▓▓▓▓▓▓▓▓▓█▓▓▓▓██████████▓▓▓█▓██▓▓▓▓▓▓█▓▓▓▓▓▓▓▓▓▓▓▓█▓█████████████▓▓▓
▒▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▒▒▒▓▓▓▓▓▓▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▓▒▒▓▓▒▒▒▒▒▒▒▒▒▒▒▓▓▒▒▒▒▒▓▓▓▓▓█▓▓▓▓▓▓██▓█▓███▓▓▓▓████▓█████████████▓▓▓▓▓▓▓▓▓██████████▒
▒▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▒▒▓▓▓▓▓▓▓▓▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▓▓▓▓▓▓▒▒▒▒▒▒▒▒▒▒▒▓▓▓▒▓▓▓▓▓▓██▓▓▓▓▓▓▓█▓███▓█▓▓█▓▓████▓▓██████████████▓▓▓▓▓█▓███████▒
▒▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▓▓▓▓▒▒▒▒▒▒▒▒▒▓▓▒▒▒▒▓▓▓▓▓▓▓▓███████████████████▓▓██████████████████▓▓▓▓▓▓▓▓▓█████▓
▒██▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▒▓▓▒▓▓▓▓▓▓▓▓▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒░░▒▒▒▒▒▒▒▓▒▒▓▓▓▓▓▓▓▓▓▒▓█▓▓▓▓▓▓▓███▓▓████████████████████████████▓██▓▓▓▓███████▓   
"""


setup(
    name="SCHISM",
    version="0.2.5",
    author="Florent Brondolo",
    author_email="florent.brondolo@akkodis.com",
    description="Simple framework for computer vision",
    packages_dir={"": "classes"},
    packages=find_packages(where="classes"),
    install_requires=[
        "torchmetrics>=1.6.1",
        "numpy>=1.17.4",
        "torch>=2.3.0",
        "matplotlib",
        "tqdm",
        "torchvision",
        "patchify",
        "peft>=0.10.0",
        "transformers>=4.40.0",
        "bitsandbytes>=0.42.0"
    ],
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: Apache-2.0 license",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.9",
)
