demo 数据为完整数据的前10条，完整数据有24000条。

### 数据说明

| 数字字段   | 具体含义                     | 数据字段   | 具体含义                                                     |
| ---------- | ---------------------------- | ---------- | ------------------------------------------------------------ |
| @timestamp | 时间戳                       | mid        | 邮件内投最终的mid                                            |
| attach     | 邮件附件名列表               | rcpt       | 收件人email列表                                              |
| authuser   | 是否为本站用户               | recviplist | 信头的Received:提取的IP地址列表                              |
| content    | 邮件内容(前512字节)          | regionip   | 从信头提取的X-Orginal-IP之类的原始的发信人IP(而不是服务器IP) |
| doccontent | Office类型邮件附件的文档信息 | sender     | 发信人                                                       |
| dwlistcnt  | 命中域名自动白名单的rcpt数量 | subject    | 邮件的主题                                                   |
| from       | 信头的from信息               | url        | 邮件中包含的URL链接(可能包含手机号码/qq号码和其他一些非URL信息) |
| fromname   | 信头的fromname               | wlistcnt   | 命中自动白名单的rcpt数量                                     |
| htmltag    | 邮件包含的html tag           | xmailer    | 信头xmailer信息                                              |
| ip         | 连接到Coremail 服务器的IP    | licenseid  | 请求的客户端的licenseid                                      |
| region     | regionip的地理位置           |            |                                                              |
