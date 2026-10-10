# تثبيت خادم العرض الوهمي
!apt-get update && apt-get install -y xvfb

# تثبيت المكتبات المعتمدة (النسخ الحديثة تدعم gymnasium)
!pip install stable-baselines3[extra] gymnasium[mujoco] pyvirtualdisplay moviepy


import os
import numpy as np
import gymnasium as gym
from stable_baselines3 import TD3
from stable_baselines3.common.noise import NormalActionNoise
from pyvirtualdisplay import Display

# 1. التأكد من تشغيل الشاشة الوهمية (مهم لكاجل)
virtual_display = Display(visible=0, size=(1400, 900))
virtual_display.start()

# 2. إنشاء البيئة
env_id = "Reacher-v4"
env = gym.make(env_id)

# 3. هندسة الضجيج (Action Noise) لضمان الاستكشاف
# نحتاج أن نعرف كم عدد مفاصل الروبوت لتوليد ضجيج لكل مفصل
n_actions = env.action_space.shape[-1]
# نصنع ضجيجاً بمقدار 10% (0.1) من قوة المحرك 
action_noise = NormalActionNoise(mean=np.zeros(n_actions), sigma=0.1 * np.ones(n_actions))

print(f"🚀 بدء بناء وتدريب نموذج TD3 لبيئة {env_id}...")

# 4. بناء نموذج TD3
model = TD3(
    "MlpPolicy", 
    env,
    action_noise=action_noise,  # حقن الضجيج الذي صنعناه
    learning_rate=0.001,        # سرعة التعلم
    buffer_size=100000,         # حجم ذاكرة الخبرة (Replay Buffer)
    batch_size=100,             # حجم الدفعة للتدريب
    tau=0.005,                  # معدل التحديث الناعم للشبكة (Soft Update)
    gamma=0.99,                 # النظرة المستقبلية
    verbose=1
)

# 5. التدريب لمدة 50,000 خطوة (تكفي جداً لهذه البيئة لتبدأ الذراع في الوصول للهدف)
model.learn(total_timesteps=50000)

# 6. حفظ النموذج
model.save("td3_reacher")
env.close()
print("✅ تم التدريب وحفظ النموذج بنجاح.")




# ==========================================
# تصوير الفيديو للتقييم بعد التطور
# ==========================================


import imageio
from IPython.display import Video, display
from stable_baselines3 import TD3

# 1. إنشاء بيئة الاختبار
env_test = gym.make("Reacher-v4", render_mode="rgb_array")
# نحدد عدد الإطارات في الثانية لنتجنب أي أخطاء
env_test.metadata["render_fps"] = 30 

# 2. تحميل النموذج
model = TD3.load("td3_reacher")

frames = []
episodes_to_record = 4  # سنسجل 4 أهداف مختلفة (حوالي 200 خطوة)
episodes_completed = 0

state, _ = env_test.reset()

print("🎥 بدء تسجيل فيديو الذراع الآلية...")

# 3. حلقة التصوير
while episodes_completed < episodes_to_record:
    frames.append(env_test.render())
    
    # نطلب من النموذج اتخاذ القرار بحتمية (بدون ضجيج استكشاف)
    action, _states = model.predict(state, deterministic=True)
    
    next_state, reward, terminated, truncated, _ = env_test.step(action)
    
    # في Reacher، المحاولة تنتهي بسبب البتر (Truncated) بعد 50 خطوة
    if terminated or truncated:
        episodes_completed += 1
        state, _ = env_test.reset()  # إعادة تعيين البيئة لظهور هدف جديد
    else:
        state = next_state

env_test.close()

# 4. حفظ الفيديو باستخدام طريقتك الموثوقة
video_path = './mujoco_reacher_td3.mp4'
imageio.mimsave(video_path, frames, fps=30, macro_block_size=None)
print(f"✅ تم تسجيل الأداء بنجاح! عدد الإطارات: {len(frames)}")

# عرض الفيديو
display(Video(video_path, embed=True))