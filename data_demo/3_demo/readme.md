demo 数据为完整数据的前10条，完整数据有790083条。

### 数据字段说明

每一行为json格式的数据，是一封邮件。

- rcpt：收户信人邮箱地址
- sender：发件人邮箱地址
- ip：企业邮箱用户登录ip
- fromname：发信人名称
- url：直接从邮件正文、subject、附件、fromname中提取出来的url 
- @timestamp：时间戳
- region：企业邮箱用登录ip所在地区
- authuser：收信时False，发信时True，注意企业邮箱域内互发的话是只有一条发信记录
- tag：邮件编号，提交答案需要用到

### 数据匿名化处理说明

#### 邮箱地址（sender, rcpt）：
[salt+hash]@[salt+hash替换掉敏感域名，其它保留].[真实TLD]
#### 示例：


```
zhangshanfeng@secret.com -> 【2585c18f43c49a78】@【ac59075b964b0715】.com

123456789@qq.com ->
2585c18f43c49a78@qq.com
```


#### fromname：

对于白名单关键词（比如admin，hr，管理员，经理等）进行保留。除了白名单的其它部分salt+hash
#### 示例：
```
马云admin ->【52d04dc20036dbd8】 admin
张帅hr    -> 【ea8a706c4c34a168】 hr
张山丰    -> 【49ba59abbe56e057】
```

#### url:

`[真实协议]://hash+salt替换掉敏感域名.真实TLD/真实参数`
#### 示例：
```
https://www.secret.com/zhangshanfeng.html ->
https://www.【ac59075b964b0715】.com/zhangshanfeng.html


http://www.baidu.com/123/456.html -> http://www.baidu.com/123/456.html
```

#### ip
经过映射处理，不是真实的ip地址，但保留子网关系
