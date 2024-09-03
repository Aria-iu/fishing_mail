import jieba

words = jieba.cut(
    "  Blocked Incoming Messages\r\n\r\n \r\n\r\n The following messages have been blocked by your administrator due to validation error.  \r\n\r\nYou have been 10 new messages in your email quarantine. Date: 21/10/2018 01:22:00 -0800 (CDT) User:  zhanghp@lzu.edu.cn  \r\n \r\n \r\n\r\nClick On Release, to Release these message(s) to your inbox folder: Deliver Messages  \r\n  Quarantined email Recipient:Subject:Date:Releasezhanghp@lzu.edu.cnFwd: MT 103 SWIFT from INFO@.... [ANZ]21/10/2019  Releasezhanghp@lzu.edu.cnSHIPMENT ARRIVAL NOT",
    cut_all = False
)
print(list(words))