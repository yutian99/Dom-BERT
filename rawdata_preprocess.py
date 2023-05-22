import re
import random
import numpy as np
from sklearn import preprocessing
import matplotlib.pyplot as plt
time=10
length=5
node2indexdict={}
index2nodedict={}
node2neibour_cnt_dict={}

class Record:
    strlist=["RecordTime","LinkID","InterfaceID","CardID","DeviceID","DNSClientIPLower","DNSClientIPHigher","DNSServerIPLower","DNSServerIPHigher","DNSReqBytes","DNSRspBytes","DNSBeginTime","DNSEndTime","DNSRspTime","Domain","Rsperrorcode","RspStatus","retranPktount","retranbytes","AppID","ConfID","FlowID","queryType","DNSClientMac","SrcPort","DstPort","DnsIpv4","recursionFlag","DnClass","req_flag_opcode","rsp_flag_aa","rsp_flag_tc","answer_rrs","authority_rrs","additional_rrs","answer_info","author_info","add_info"]
    raw_list=[]
    def __init__(self,raw_list):
        self.raw_list=raw_list
        self.RecordTime=raw_list[0]
        self.LinkID=raw_list[1]#
        self.InterfaceID=raw_list[2]#
        self.CardID=raw_list[3]#
        self.DeviceID=raw_list[4]#
        self.DNSClientIPLower=raw_list[5]
        self.DNSClientIPHigher=raw_list[6]
        self.DNSServerIPLower=raw_list[7]
        self.DNSServerIPHigher=raw_list[8]
        self.DNSReqBytes=raw_list[9]
        self.DNSRspBytes=raw_list[10]
        self.DNSBeginTime=raw_list[11]
        self.DNSEndTime=raw_list[12]
        self.DNSRspTime=raw_list[13]
        self.Domain=raw_list[14]
        if self.Domain[0]=="\"":
            self.Domain=self.Domain[1:]
        if self.Domain[-1]=="\"":
            self.Domain=self.Domain[:-1]
        self.Rsperrorcode=raw_list[15]
        self.RspStatus=raw_list[16]
        self.retranPktount=raw_list[17]
        self.retranbytes=raw_list[18]
        self.AppID=raw_list[19]#
        self.ConfID=raw_list[20]#
        self.FlowID=raw_list[21]#
        self.queryType=raw_list[22]
        self.DNSClientMac=raw_list[23]#
        self.SrcPort=raw_list[24]
        self.DstPort=raw_list[25]
        self.DnsIpv4=raw_list[26]#
        self.recursionFlag=raw_list[27]
        self.DnClass=raw_list[28]
        self.req_flag_opcode=raw_list[29]
        self.rsp_flag_aa=raw_list[30]
        self.rsp_flag_tc=raw_list[31]
        self.answer_rrs=raw_list[32]
        self.authority_rrs=raw_list[33]
        self.additional_rrs=raw_list[34]
        self.answer_info=raw_list[35]
        self.author_info=raw_list[36]
        self.add_info=raw_list[37]
    def display(self):
        for i in range(len(self.strlist)):
            print(self.strlist[i]+":"+self.raw_list[i])
def list2str(mylist):
    rtnstr=""
    for i in mylist:
        rtnstr=rtnstr+str(i)+" "
    return rtnstr.strip()

def transferip(inputip):
    ip=bin(int(inputip))[2:]
    zero="0"*(32-len(ip))
    ip=zero+ip
    outputip=[]
    outputip.append(int(ip[:8],2))
    outputip.append(int(ip[8:16],2))
    outputip.append(int(ip[16:24],2))
    outputip.append(int(ip[24:],2))
    return outputip

    




def get_random_walk():
    with open("./dataset/ablation_did+dd.txt","w") as fout:
        walk_L=50
        walk_n=10
        for i in range(walk_n):
            print(i)
            for node in node2indexdict:
                #if node.ntype[0]=='d' and len(node.getTotalAdj()) >0:
                if node.ntype[0] == 'd' and len(node.adj_i)+len(node.adj_d) > 0:
                    walk_list=[]
                    cur_node=node
                    while len(walk_list)<walk_L:
                        rand_p = random.random()
                        if rand_p > 0.5:
                            #nodes_list = cur_node.getTotalAdj()
                            nodes_list = list(cur_node.adj_i)+list(cur_node.adj_d)
                            tmp_node = index2nodedict[random.choice(nodes_list)]
                            iter=0
                            while len(tmp_node.adj_d) ==0:
                                #if tmp_node.ntype[0] == 'd':
                                    #walk_list.append(tmp_node.ntype[0]+str(tmp_node.num))
                                tmp_node = index2nodedict[random.choice(nodes_list)]
                                iter+=1
                            nodes_list = list(tmp_node.adj_d)
                            tmp_node = index2nodedict[random.choice(nodes_list)]
                            cur_node=tmp_node
                            if cur_node.ntype[0]=='d':
                                walk_list.append(cur_node.ntype[0]+str(cur_node.num))
                        else:
                            cur_node=node

                    fout.write(node.ntype[0]+str(node.num)+":"+" ".join(walk_list)+"\n")

def get_prob():
    with open("./dataset/ablation_did+dd.txt", "r") as fin:
        inputstr = fin.readline()
        while inputstr:
            keynode,neibour_list = inputstr.strip().split(":")[0],inputstr.strip().split(":")[1].split(" ")
            if keynode not in node2neibour_cnt_dict:
                node2neibour_cnt_dict[keynode]={}
            for neibour in neibour_list:
                if neibour not in node2neibour_cnt_dict[keynode]:
                    node2neibour_cnt_dict[keynode][neibour]=1
                else:
                    node2neibour_cnt_dict[keynode][neibour]+=1
            inputstr = fin.readline()
    for node in node2neibour_cnt_dict:
        values=list(node2neibour_cnt_dict[node].values())
        maxval=max(values)
        for neibour  in node2neibour_cnt_dict[node]:
            node2neibour_cnt_dict[node][neibour]=node2neibour_cnt_dict[node][neibour]/maxval/2+0.5
def dataset_build():
    mal_set = set()
    ben_set = set()
    mal_node_set=set()
    ben_node_set=set()
    same_num=50
    diff_num=50
    hist_list=[]
    with open("./dataset/d_labels.txt", "r") as fin:
        for line in fin.readlines():
            line = line.strip()
            l = int(line[-1])
            if l == 1:
                mal_set.add(line[:-2])
            elif l==0:
                ben_set.add(line[:-2])
    for nodeidx in node2neibour_cnt_dict:
        node=index2nodedict[nodeidx]
        if node.value in mal_set:
            mal_node_set.add(node)
        elif node.value in ben_set:
            ben_node_set.add(node)
    print("malicious domain num:{}\nbenign domain num:{}".format(len(mal_node_set),len(ben_node_set)))
    with open("./dataset/data_ablation_did+dd.txt", "w") as fout:
        for node in mal_node_set:
            node_i="d"+str(node.num)
            chosen=[]
            if node_i not in node2neibour_cnt_dict:
                print(node_i)
                continue
            neibour_order=sorted(node2neibour_cnt_dict[node_i].items(), key=lambda x: x[1], reverse=True)
            for nodeidx,prob in neibour_order:
                same_num = 50
                diff_num = 50
                if index2nodedict[nodeidx].value in mal_set and same_num>0:
                    same_num-=1
                    chosen.append(nodeidx)
                    hist_list.append(prob)
                    fout.write(node.value+" "+ index2nodedict[nodeidx].value+" "+str(prob)[:6]+"\n")
                if index2nodedict[nodeidx].value in ben_set and diff_num>0:
                    diff_num-=1
                    chosen.append(nodeidx)
                    hist_list.append(prob)
                    fout.write(node.value+" "+index2nodedict[nodeidx].value+" "+str(prob)[:6]+"\n")
            while same_num>0:
                n=random.choice(list(mal_node_set))
                nidx="d"+str(n.num)
                if nidx not in chosen:
                    chosen.append(nidx)
                    fout.write(node.value+" " + index2nodedict[nidx].value+" " + "0.5\n")
                    hist_list.append(0.5)
                    same_num-=1
            while diff_num>0:
                n=random.choice(list(ben_node_set))
                nidx = "d" + str(n.num)
                if nidx not in chosen:
                    chosen.append(nidx)
                    fout.write(node.value+" " + index2nodedict[nidx].value+" " + "0\n")
                    hist_list.append(0)
                    diff_num-=1
        for node in ben_node_set:
            node_i = "d" + str(node.num)
            if node_i not in node2neibour_cnt_dict:
                print(node_i)
                continue
            chosen=[]
            neibour_order=sorted(node2neibour_cnt_dict[node_i].items(), key=lambda x: x[1], reverse=True)
            for nodeidx,prob in neibour_order:
                same_num = 50
                diff_num = 50
                if nodeidx in mal_set and same_num>0:
                    same_num-=1
                    chosen.append(nodeidx)
                    hist_list.append(prob)
                    fout.write(node.value+" "+index2nodedict[nodeidx].value+" "+str(prob)[:6]+"\n")
                if nodeidx in ben_set and diff_num>0:
                    diff_num-=1
                    chosen.append(nodeidx)
                    hist_list.append(prob)
                    fout.write(node.value+" "+index2nodedict[nodeidx].value+" "+str(prob)[:6]+"\n")
            while same_num>0:
                n=random.choice(list(mal_node_set))
                nidx = "d" + str(n.num)
                if nidx not in chosen:
                    chosen.append(nidx)
                    hist_list.append(0.5)
                    fout.write(node.value+" " + index2nodedict[nidx].value+" " + "0.5\n")
                    same_num-=1
            while diff_num>0:
                n=random.choice(list(ben_node_set))
                nidx = "d" + str(n.num)
                if nidx not in chosen:
                    chosen.append(nidx)
                    hist_list.append(0)
                    fout.write(node.value+" " + index2nodedict[nidx].value+" " + "0\n")
                    diff_num-=1
    #plt.hist(hist_list)
    #plt.show()
class Node:
    def __init__(self,ntype,value,num):
        self.ntype=ntype #i,d,c
        self.value=value
        self.num=num
        self.adj_i=set()
        self.adj_c=set()
        self.adj_d=set()
    def __eq__(self, other):#比较两个对象是否相等的函数
        return self.value == other.value and self.ntype==other.ntype
    def __hash__(self):
        return hash(self.value)+hash(self.ntype)
    def addAdj(self,ntype,num):
        if ntype=='ip':
            self.adj_i.add(num)
        elif ntype=='domain':
            self.adj_d.add(num)
        else:
            self.adj_c.add(num)
    def deleteAdj(self,ntype,num):
        if ntype=='ip':
            self.adj_i.remove(num)
        elif ntype=='domain':
            self.adj_d.remove(num)
        else:
            self.adj_c.remove(num)
    def getTotalAdj(self):
        rtn_list=[]
        for item in self.adj_c:
                rtn_list.append(str(item))
        for item in self.adj_d:
                rtn_list.append(str(item))
        for item in self.adj_i:
            rtn_list.append(str(item))
        return rtn_list

with open("./dataset/raw_data.txt","r") as fin:
    inputstr=fin.readline()
    records=[]
    clientset=set()
    deviceset=set()
    linkset=set()
    macset=set()
    domainset=set()
    flowset=set()
    DnsIpv4set=set()
    ipset=set()
    
    while inputstr:
        raw_list=inputstr.strip().split(", ")
        records.append(Record(raw_list))
        inputstr=fin.readline()
    #'''
    suc_count=0
    a_count=0
    client_count=-1
    domain_count=-1
    ip_count=-1
    for i in range(len(records)):
        if records[i].RspStatus=='2':
            suc_count+=1
            tmpnode=Node('client',records[i].DNSClientIPLower,client_count+1)
            if tmpnode in node2indexdict:
                from_client_node=node2indexdict[tmpnode]
            else:
                client_count += 1
                index="c"+str(client_count)
                node2indexdict[tmpnode]=index
                from_client_node=index
                index2nodedict[index]=tmpnode
            clientset.add(records[i].DNSClientIPLower)
            deviceset.add(records[i].DeviceID)
            macset.add(records[i].DNSClientMac)
            linkset.add(records[i].LinkID)
            flowset.add(records[i].FlowID)
            
            if records[i].queryType=='1'  and records[i].answer_rrs!='0':
                #records[num，（type,name,ttl）,...]
                #query
                domainset.add(records[i].Domain)
                tmpnode=Node('domain',records[i].Domain,domain_count+1)
                if tmpnode in node2indexdict:
                    query_domain_node=node2indexdict[tmpnode]
                else:
                    domain_count += 1
                    index = "d" + str(domain_count)
                    node2indexdict[tmpnode]=index
                    query_domain_node=index
                    index2nodedict[index]=tmpnode
                index2nodedict[from_client_node].addAdj('domain',query_domain_node)
                index2nodedict[query_domain_node].addAdj('client',from_client_node)
                s='\[\d+,(\(.*\))\]'
                line=records[i].answer_info
                m=re.match(s,line)
                if m is None:
                    #print(i)
                    pass
                else:
                    s1=m.group(1)
                    pattern=re.compile('\(.*?\)')
                    result=pattern.findall(s1)
                    for record in result:
                        record=record[1:-2]
                        record=record.split(",")
                        rtype,value,ttl=record
                        if rtype=='CNAME':
                            domainset.add(value)
                            tmpnode=Node('domain',value,domain_count+1)
                            if tmpnode in node2indexdict:
                                domain_cname_node=node2indexdict[tmpnode]
                            else:
                                domain_count += 1
                                index = "d" + str(domain_count)
                                node2indexdict[tmpnode]=index
                                domain_cname_node=index
                                index2nodedict[index]=tmpnode
                                #cname
                            index2nodedict[query_domain_node].addAdj('domain',domain_cname_node)
                            index2nodedict[domain_cname_node].addAdj('domain',query_domain_node)
                        elif rtype=='A' or rtype=='AAAA':
                            ipset.add(value)
                            tmpnode=Node('ip',value,ip_count+1)
                            if tmpnode in node2indexdict:
                                ip_map_node=node2indexdict[tmpnode]
                            else:
                                ip_count+=1
                                index = "i" + str(ip_count)
                                node2indexdict[tmpnode]=index
                                ip_map_node=index
                                index2nodedict[index]=tmpnode
                            index2nodedict[query_domain_node].addAdj('ip',ip_map_node)
                            index2nodedict[ip_map_node].addAdj('domain',query_domain_node)
                            #map
                a_count+=1
                domainset.add(records[i].Domain)
                DnsIpv4set.add(records[i].DnsIpv4)

            else:
                domainset.add(records[i].Domain)
                tmpnode = Node('domain', records[i].Domain, domain_count + 1)
                if tmpnode in node2indexdict:
                    query_domain_node = node2indexdict[tmpnode]
                else:
                    #domain_f.write(records[i].Domain+"\n")
                    domain_count += 1
                    index = "d" + str(domain_count)
                    node2indexdict[tmpnode] = index
                    query_domain_node = index
                    index2nodedict[index] = tmpnode
                index2nodedict[from_client_node].addAdj('domain', query_domain_node)
                index2nodedict[query_domain_node].addAdj('client', from_client_node)

        else:
            clientset.add(records[i].DNSClientIPLower)
            tmpnode=Node('client',records[i].DNSClientIPLower,client_count+1)
            if tmpnode in node2indexdict:
                from_client_node=node2indexdict[tmpnode]
            else:
                client_count += 1
                index = "c" + str(client_count)
                node2indexdict[tmpnode] = index
                from_client_node = index
                index2nodedict[index] = tmpnode

            if records[i].queryType=='1':
                domainset.add(records[i].Domain)
                tmpnode=Node('domain',records[i].Domain,domain_count+1)
                if tmpnode in node2indexdict:
                    query_domain_node=node2indexdict[tmpnode]

                else:
                    domain_count += 1
                    index = "d" + str(domain_count)
                    node2indexdict[tmpnode] = index
                    query_domain_node = index
                    index2nodedict[index] = tmpnode

                index2nodedict[from_client_node].addAdj('domain',query_domain_node)
                index2nodedict[query_domain_node].addAdj('client',from_client_node)    


    #'''





print("size:"+str(len(clientset)))

prune_ip_set=set()
prune_client_set=set()
prune_domain_set=set()
#print(len(node2indexdict))
for node in node2indexdict:
    if node.ntype=='ip' and len(node.adj_d)==1:
        prune_ip_set.add(node)
    elif node.ntype=='client' and len(node.adj_d)<=2:
        prune_client_set.add(node)
for node in prune_ip_set:
    del node2indexdict[node]
    del index2nodedict['i'+str(node.num)]
    for item in node.adj_d:
        index2nodedict[str(item)].deleteAdj('ip','i'+str(node.num))
    for item in node.adj_c:
        index2nodedict[str(item)].deleteAdj('ip','i'+str(node.num))
    for item in node.adj_i:
        index2nodedict[str(item)].deleteAdj('ip','i'+str(node.num))
for node in prune_client_set:
    del node2indexdict[node]
    del index2nodedict['c' + str(node.num)]
    for item in node.adj_d:
        index2nodedict[str(item)].deleteAdj('client','c'+str(node.num))
    for item in node.adj_c:
        index2nodedict[str(item)].deleteAdj('client','c'+str(node.num))
    for item in node.adj_i:
        index2nodedict[str(item)].deleteAdj('client','c'+str(node.num))
for node in node2indexdict:
    if node.ntype=='domain' and len(node.adj_c)<=1 and len(node.adj_d)==0 and len(node.adj_i)<=1:
        prune_domain_set.add(node)
for node in prune_domain_set:
    del node2indexdict[node]
    '''
    del index2nodedict['d' + str(node.num)]
    for item in node.adj_d:
        index2nodedict[str(item)].deleteAdj('domain','d'+str(node.num))
    for item in node.adj_c:
        index2nodedict[str(item)].deleteAdj('domain','d'+str(node.num))
    for item in node.adj_i:
        index2nodedict[str(item)].deleteAdj('domain','d'+str(node.num))
    '''

print("total node num:{}".format(len(node2indexdict)))

'''
with open("./dataset/domain_id2name.txt", "w") as fout:
    for node in node2indexdict:
        if node.ntype[0] == 'd':
            fout.write(str(node.num)+":"+node.value+"\n")
'''


#get_random_walk()
#get_prob()
#dataset_build()
