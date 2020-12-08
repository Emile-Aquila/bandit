use std::io;
use rand::Rng;
use std::f32;

struct bandit{
    arm_size: usize,
    prob:Vec<f32>,
}

impl bandit{
    fn get_rew(&self,a:usize) -> f32{
        let mut rng = rand::thread_rng();
        let tmp :f32 = rng.gen();
        if tmp < self.prob[a] { 1.0 }else { 0.0 }
    }
}

struct epsilon_agent{
    arm_size: usize,
    eps: f32,
    tries:Vec<f32>,
    score:Vec<f32>,
    score_all:f32,
    try_all:f32,
}

impl epsilon_agent{
    fn get_max(&self)->usize{
        let mut ans:usize = 0;
        let mut mx = 0.0;
        for (i, &val) in self.score.iter().enumerate() {
            if mx < val {
                mx = val;
                ans = i;
            }
        }
        ans
    }
    fn rand_act(&self)->usize{
        let mut rng = rand::thread_rng();
        let mut act:usize = rng.gen();
        act % self.arm_size
    }
    fn policy(&self)->usize{
        let mut rng = rand::thread_rng();
        let tmp:f32 = rng.gen();
        if tmp >= self.eps{
            self.get_max()
        }else {
            self.rand_act()
        }
    }
    fn act(&mut self,band:&bandit){
        let action = self.policy();
        let rew = band.get_rew(action);
        self.score_all += rew;
        self.try_all += 1.0;
        self.score[action] = (self.score[action] * self.tries[action] + rew)/(self.tries[action]+1.0);
        self.tries[action] += 1.0;
    }
}

struct softmax_agent{
    arm_size: usize,
    tries:Vec<f32>,
    score:Vec<f32>,
    score_all:f32,
    try_all:f32,
    beta:f32,
}

impl softmax_agent{
    fn sum(&self)->f32{
        let mut ans = 0.0;
        for (_, &val) in self.score.iter().enumerate() {
            ans += (val*self.beta).exp();
        }
        ans
    }
    fn policy(&self)->usize{
        let sum = self.sum();
        let mut tmp = 0.0;
        let mut rng = rand::thread_rng();
        let flg:f32 = rng.gen();
        for i in 0..self.arm_size {
            tmp += (self.score[i] * self.beta).exp() / sum;
            if flg <= tmp {
                return i as usize;
            }
        }
        6 //otherwise,例外
    }
    fn act(&mut self,band:&bandit){
        let action = self.policy();
        let rew = band.get_rew(action);
        self.score_all += rew;
        self.try_all += 1.0;
        self.score[action] = (self.score[action] * self.tries[action] + rew)/(self.tries[action]+1.0);
        self.tries[action] += 1.0;
    }
}


struct optimistic_agent{
    arm_size: usize,
    tries:Vec<f32>,
    score:Vec<f32>,
    score_all:f32,
    try_all:f32,
}

impl optimistic_agent{//UCB1
    fn q(&self,a:usize)->f32{ self.score[a] }
    fn u(&self,a:usize)->f32{
        let mut ans = self.try_all.ln()*2.0;
        ans /= self.tries[a];
        ans.sqrt()
    }
    fn policy(&self)->usize{
        let mut ans:usize = 0;
        let mut mx = 0.0;
        for i in 0..self.arm_size{
            if mx < self.q(i) + self.u(i) {
                mx = self.q(i) + self.u(i);
                ans = i;
            }
        }
        ans
    }
    fn act(&mut self,band:&bandit,action:usize){
        let rew = band.get_rew(action);
        self.score_all += rew;
        self.try_all += 1.0;
        self.score[action] = (self.score[action] * self.tries[action] + rew)/(self.tries[action]+1.0);
        self.tries[action] += 1.0;
    }
}

fn main(){
    let bandit_machine = bandit{
        arm_size: 5,
        prob:vec![0.1, 0.2, 0.4, 0.5, 0.7],
    };
    let mut eps_agent = epsilon_agent{
        arm_size: 5,
        eps: 0.1,
        tries:vec![0.0; 5],
        score:vec![0.0; 5],
        score_all:0.0,
        try_all:0.0,
    };
    let mut sft_agent = softmax_agent{
        arm_size: 5,
        tries:vec![0.0; 5],
        score:vec![0.0; 5],
        score_all:0.0,
        try_all:0.0,
        beta:10.0,
    };
    let mut opt_agent = optimistic_agent{
        arm_size: 5,
        tries:vec![0.0; 5],
        score:vec![0.0; 5],
        score_all:0.0,
        try_all:0.0,
    };

    let rep_time = 100.0;
    let flg = 0.67;

    while eps_agent.try_all < rep_time || eps_agent.score_all/eps_agent.try_all < flg{
        eps_agent.act(&bandit_machine);
    }
    //epsilon-greedy
    println!("--------------eps-greedy------------------------");
    println!("time step                  {}",eps_agent.try_all);
    println!("try times of each arms     {:?}", eps_agent.tries);
    println!("average score of each arms {:?}", eps_agent.score);
    println!("average score               {}",eps_agent.score_all/eps_agent.try_all);

    //softmax
    while sft_agent.try_all < rep_time || sft_agent.score_all/sft_agent.try_all < flg{
        sft_agent.act(&bandit_machine);
    }
    println!("--------------softmax------------------------");
    println!("time step                  {}",sft_agent.try_all);
    println!("try times of each arms     {:?}", sft_agent.tries);
    println!("average score of each arms {:?}", sft_agent.score);
    println!("average score               {}",sft_agent.score_all/sft_agent.try_all);


    //optimistic-agent
    for i in 0..opt_agent.arm_size{
        opt_agent.act(&bandit_machine,i);
    }
    while opt_agent.try_all < rep_time || opt_agent.score_all/opt_agent.try_all < flg {
        opt_agent.act(&bandit_machine,opt_agent.policy());
    }
    println!("--------------ucb1------------------------");
    println!("time step                  {}",opt_agent.try_all);
    println!("try times of each arms     {:?}", opt_agent.tries);
    println!("average score of each arms {:?}", opt_agent.score);
    println!("average score               {}",opt_agent.score_all/opt_agent.try_all);
}
