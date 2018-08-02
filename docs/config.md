# 配置文件config.py
该文件指定了用于fast rcnn训练的默认config选项，不能随意更改，
如需更改，应当用yaml再写一个config_file，然后使用cfg_from_file(filename)导入以覆盖默认config。

cfg_from_file(filename)定义见该文件。
tools目录下的绝大多数文件采用--cfg 选项来指定重写的配置文件（默认采用默认的config）

# 参考
http://www.cnblogs.com/alanma/p/6800944.html