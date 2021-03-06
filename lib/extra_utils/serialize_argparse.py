# -----------------------------------------------------
# -*- coding: utf-8 -*-
# @Time    : 8/9/2018 4:30 PM
# @Author  : sunyonghai
# @Software: ZJ_AI
# -----------------------------------------------------
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import os
import json
import argparse

from morelib.utils import io_utils
from model.config import cal_data_aug_code


class SerializeArgparse:
    def __init__(self, description):
        self.parser = argparse.ArgumentParser(description=description)
        self.loaded = {}
        self.args = {}
        self.parsed = None

    def add_argument(self, name, type, dest="", default=None, help="", save=True):
        self.parser.add_argument(name, type=type, dest=dest, default=None, help=help)
        if name[0] == '-':
            name = name[1:]

        self.args[dest] = {
            "type": type,
            "dest": dest,
            "default": default,
            "save": save
        }

    def do_parse_args(self, loaded={}):
        self.parsed = self.parser.parse_args()
        for k, v in self.parsed.__dict__.items():
            if v is None:
                if k in loaded and self.args[k]["save"]:
                    self.parsed.__dict__[k] = loaded[k]
                else:
                    self.parsed.__dict__[k] = self.args[k]["default"]
        return self.parsed

    def parse_or_cache(self):
        if self.parsed is None:
            self.do_parse_args()

    def parse_args(self):
        self.parse_or_cache()
        return self.parsed

    def save(self, fname):
        self.parse_or_cache()
        with open(fname, 'w') as outfile:
            json.dump(self.parsed.__dict__, outfile, indent=4)
            return True

    def load(self, fname):
        if os.path.isfile(fname):
            data = {}
            with open(fname, "r") as data_file:
                data = json.load(data_file)

            self.do_parse_args(data)
        return self.parsed

    def reload_or_save_args(self,cfg,args):
        if args.load_args_json:
            args = self.load(os.path.join(cfg.ROOT_DIR, args.args_json_dir))
        else:
            if args.tag == None:
                data_name = 'default'
            else:
                data_name = args.tag
            postfix = cal_data_aug_code(cfg)

            args_parse_path = os.path.join(cfg.ROOT_DIR, 'data', 'args_parse', cfg.EXP_DIR, data_name)
            io_utils.mkdir(args_parse_path)

            self.save(os.path.join(args_parse_path, "{}_{}.json".format("+".join(args.package_name), postfix)))
            print("args save in {}".format(os.path.join(args_parse_path, "{}_{}.json".format("+".join(args.package_name), postfix))))
        return args