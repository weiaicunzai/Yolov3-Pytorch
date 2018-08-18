""" settings and configurations for yolo

Read value from global_settings.py
"""

import conf.global_settings 


class Settings:
    def __init__(self, settings_modules):

        #dynamic construct attributes
        for attr in dir(settings_modules):
            if attr.isupper():
                setattr(self, attr, getattr(settings_modules, attr))



settings = Settings(conf.global_settings)