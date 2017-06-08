# python笔记
### 5.25

``` python
import json
path = 'ch02/usagov_bitly_data2012-03-16-1331923249.txt'

records = [json.loads(line) for line in open(path)]
%pwd
path = 'ch02/usagov_bitly_data2012-03-16-1331923249.txt'
open(path).readline()

records[0]
records[0]['tz']
print(records[0]['tz'])

print(records[0]['tz'])

```





