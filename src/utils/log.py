import logging


'''使用logging的流程
'''

def setuplogging(logfile='0.log'):  # 默认是0.logfile， 当传入其它名字的logfile的时候，就不是0.logfile了。
    # logging.basicConfig具体是干嘛？ ：basicConfig方法是一个快速的设置方式，可以快速配置日志的输出格式、日志级别以及输出位置（控制台或文件）
    logging.basicConfig(filename=logfile,        # 记录到专门的日志文件中去。 
                            level=logging.INFO,    # 应该是等级低于info的消息不打印出来。只有warning，info，critical级别的日志才打印出来。
                            filemode='w',
                            format='%(asctime)s - %(message)s')   # 
    return
'''
format 参数常用占位符：
%(asctime)s：时间
%(name)s：logger 名称
%(levelname)s：日志级别
%(filename)s：当前文件名
%(lineno)d：行号
%(message)s：日志信息
'''

def log_args(args):   # logging这个库还可以有多个logger。    我这里用不着多个logger
    logger = logging.getLogger() # get the default logger  # 用于创建或获取一个日志记录器（Logger）对象
    for arg in vars(args):  # var用于返回对象object的属性和属性值的字典对象 。 而arg则获取字典的每一个key。
        logger.info("{0: <15} : {1:}".format(arg, getattr(args, arg)))  # ？  getattr(args, arg) 获取参数值
        # {0:<15} → 第一个参数（arg）左对齐，占 15 字符宽度  {1} → 第二个参数（参数值）  如："batch_size     : 32"
    return   # return什么？


