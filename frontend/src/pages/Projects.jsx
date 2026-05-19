import SectionPage from '../components/SectionPage'

const PROJECTS = [
  {
    name: 'GPU Architecture Simulator',
    desc: 'Built a cycle-accurate GPU SM simulator with warp scheduling, caches, and SIMT. Validated with 12 CUDA kernels, reducing prediction error from 21% to 7%.',
    tags: ['C++', 'CUDA', 'Python', 'Computer Architecture'],
    links: [{ label: 'DETAILS', href: '#' }],
  },
  {
    name: 'Multiprogramming Operating System',
    desc: 'Simulated an OS using C++ with data read/write operations, register management, paging, input/output spooling, and error handling. Achieved 30% efficiency improvement.',
    tags: ['C++', 'Operating Systems', 'Algorithms'],
    links: [{ label: 'DETAILS', href: '#' }],
  },
  {
    name: 'Advanced Game Engine Optimization',
    desc: 'Implemented frustum culling with AABB and multi-threaded Physics-Component system. Enhanced rendering performance with 100+ characters in 3D scenes.',
    tags: ['C++', 'Physics Simulation', '3D Graphics', 'Multithreading'],
    links: [{ label: 'DETAILS', href: '#' }],
  },
  {
    name: 'Stock Trading & Portfolio Management App',
    desc: 'Built full-stack application using ReactJS, Spring Boot, and MongoDB. Features stock search, watchlist manager, buy/sell capabilities, and portfolio tracking.',
    tags: ['React.js', 'Spring Boot', 'MongoDB', 'Java'],
    links: [{ label: 'GITHUB', href: '#' }, { label: 'DETAILS', href: '#' }],
  },
  {
    name: 'µSocial - Event-Driven Microservices Architecture',
    desc: 'Production-grade microservices ecosystem implementing a scalable social media platform using event sourcing, CQRS, and polyglot persistence. Demonstrates Domain-Driven Design with Apache Kafka, PostgreSQL, MongoDB, Neo4j, and Redis.',
    tags: ['Kafka', 'CQRS', 'Event Sourcing', 'PostgreSQL', 'MongoDB', 'Kubernetes', 'Istio'],
    links: [{ label: 'DETAILS', href: '#' }],
  },
  {
    name: 'Event Booking and Management App',
    desc: 'Designed a responsive full-stack application using Angular, NodeJS, and MongoDB with comprehensive searching and booking features. Successfully deployed on Google Cloud Platform.',
    tags: ['Angular', 'Node.js', 'MongoDB', 'GCP'],
    links: [{ label: 'DETAILS', href: '#' }],
  },
  {
    name: 'Flight Delay Prediction',
    desc: 'Developed a machine learning model to predict flight delays up to four days in advance with 65% accuracy. Analyzed historical flight data and weather patterns using feature engineering.',
    tags: ['Python', 'Machine Learning', 'Random Forest', 'Gradient Boosting'],
    links: [{ label: 'DETAILS', href: '#' }],
  },
  {
    name: 'Reselling Used Textbooks App',
    desc: 'Implemented a RESTful API in Java to handle used textbook inventory with design patterns and clean architecture. Optimized book resale operations by 30%.',
    tags: ['Java', 'REST API', 'Design Patterns', 'Clean Architecture'],
    links: [{ label: 'GITHUB', href: '#' }, { label: 'DETAILS', href: '#' }],
  },
  {
    name: 'Medical Chatbot for Symptom Analysis',
    desc: 'Designed a chatbot using advanced NLP techniques and LLMs such as GPT-4 and BERT. Incorporates text preprocessing, Naive Bayes, RNNs, and transformers. Achieved 85% accuracy.',
    tags: ['NLP', 'GPT-4', 'BERT', 'RNN', 'Python'],
    links: [{ label: 'DETAILS', href: '#' }],
  },
  {
    name: 'Generation M',
    desc: 'Engineered a web application using advanced models including Imagen, Tacotron 2, DALL-E, and Stable Diffusion to efficiently generate images and audio from text.',
    tags: ['DALL-E', 'Stable Diffusion', 'Imagen', 'Tacotron 2', 'Python'],
    links: [{ label: 'DETAILS', href: '#' }],
  },
  {
    name: 'Plagiarism Detector',
    desc: 'Engineered a robust plagiarism detection system leveraging advanced TF-IDF and vectorization methodologies. Employs sophisticated feature extraction techniques on textual data.',
    tags: ['Python', 'TF-IDF', 'NLP', 'Machine Learning'],
    links: [{ label: 'DETAILS', href: '#' }],
  },
  {
    name: 'ReadEasy',
    desc: 'Designed and constructed an online web application facilitating the purchase, sale, and rental of educational books. Built with XAMPP framework.',
    tags: ['XAMPP', 'PHP', 'MySQL', 'Web Development'],
    links: [{ label: 'DETAILS', href: '#' }],
  },
  {
    name: 'Bot Detection System in Twitter',
    desc: 'Developed a system to distinguish between real users and bot accounts on Twitter using machine learning algorithms to classify accounts based on behavioral and content-based characteristics.',
    tags: ['Machine Learning', 'Python', 'Data Mining', 'Social Media'],
    links: [{ label: 'DETAILS', href: '#' }],
  },
  {
    name: 'Headliner - Personalized News App',
    desc: 'Built a personalized news aggregation app using React.js with TailwindCSS and Node.js backend. Integrates with News API for fresh content. Features user preference settings.',
    tags: ['React.js', 'TailwindCSS', 'Node.js', 'News API'],
    links: [{ label: 'GITHUB', href: '#' }, { label: 'LIVE ↗', href: '#' }],
  },
  {
    name: 'TaskCLI - Productivity in the Terminal',
    desc: 'A CLI-based task management tool. Add tasks with priorities and due dates, filter and list with flexible parameters, mark complete, and track productivity statistics.',
    tags: ['Python', 'Click CLI', 'SQLite', 'Rich'],
    links: [{ label: 'GITHUB', href: '#' }, { label: 'DETAILS', href: '#' }],
  },
  {
    name: 'LAMP - Job Search Efficiency App',
    desc: 'A mobile app implementing the LAMP method inspired by Steve Dalton\'s 2-Hour Job Search. Features progress tracking, calendar integration, and completion statistics.',
    tags: ['React Native', 'TypeScript', 'Animated API', 'React Hooks'],
    links: [{ label: 'GITHUB', href: '#' }, { label: 'DETAILS', href: '#' }],
  },
  {
    name: 'LeetCode Problem Recommender',
    desc: 'An intelligent recommender system that adapts to developer progress. Uses cosine similarity for topic matching and personalized recommendations based on skill level and solving history.',
    tags: ['Python', 'Pandas', 'Scikit-Learn', 'Matplotlib'],
    links: [{ label: 'GITHUB', href: '#' }, { label: 'DETAILS', href: '#' }],
  },
  {
    name: 'Real-Time Object Detection with YOLO',
    desc: 'Implemented the popular YOLO algorithm to detect objects in real-time. Detects 80+ object classes with bounding boxes, confidence scores, and non-maximum suppression.',
    tags: ['Python', 'OpenCV', 'YOLOv3', 'Computer Vision'],
    links: [{ label: 'GITHUB', href: '#' }, { label: 'DETAILS', href: '#' }],
  },
  {
    name: 'ECS Research Day Automation Challenge',
    desc: 'Developed an end-to-end automation system for research poster judging including intelligent judge assignment, mobile scoring interface, and bias-corrected fair ranking algorithm.',
    tags: ['Angular', 'Supabase', 'NLP', 'Algorithm Design'],
    links: [{ label: 'GITHUB', href: '#' }, { label: 'DETAILS', href: '#' }],
  },
]

export default function Projects() {
  return (
    <SectionPage color="#a4bde8" icon="🗂️" title="PROJECTS">
      <p className="section-label">// things i built</p>

      <div className="projects-grid">
        {PROJECTS.map((p, i) => (
          <div className="project-card" key={i}>
            <div className="project-name">{p.name}</div>
            <div className="project-desc">{p.desc}</div>
            <div className="tags">
              {p.tags.map(t => <span className="tag" key={t}>{t}</span>)}
            </div>
            <div className="project-links">
              {p.links.map(l => (
                <a href={l.href} className="project-link" key={l.label}>{l.label}</a>
              ))}
            </div>
          </div>
        ))}
      </div>
    </SectionPage>
  )
}
