# 📚 ChunIVision Documentation Index

Quick navigation to all project documentation.

## 🚀 Getting Started

**New to the project? Start here:**

1. **[README.md](../README.md)** - Project overview, features, and quick start
2. **[FRAMEWORK_OVERVIEW.md](../FRAMEWORK_OVERVIEW.md)** - Visual overview of what's been created
3. **[GETTING_STARTED.md](GETTING_STARTED.md)** - 12-week implementation roadmap
4. **[IMPLEMENTATION_CHECKLIST.md](../IMPLEMENTATION_CHECKLIST.md)** - Track your progress

## 📖 Core Documentation

### Architecture & Design

- **[ARCHITECTURE.md](ARCHITECTURE.md)** (23KB)
  - System architecture overview
  - Data flow diagrams
  - Threading model
  - Component interactions
  - Error handling strategy

### API Specifications

- **[MODULE_SPECS.md](MODULE_SPECS.md)** (39KB)
  - Complete API for all 22 modules
  - Class interfaces with type hints
  - Method signatures and docstrings
  - Data structures
  - Module dependencies

### Communication Protocols

- **[OUTPUT_PROTOCOLS.md](OUTPUT_PROTOCOLS.md)** (19KB)
  - Serial port protocol (binary/text)
  - HID device specification
  - Keyboard mapping
  - UDP protocol (binary/JSON)
  - Protocol comparison and benchmarks

## 🛠️ Development Guides

### For Developers

- **[DEVELOPMENT.md](DEVELOPMENT.md)** (24KB)
  - Code style and conventions
  - Testing strategies
  - Adding new features
  - Debugging techniques
  - Best practices
  - CI/CD setup

### For Users

- **[CALIBRATION_GUIDE.md](CALIBRATION_GUIDE.md)** (19KB)
  - Camera setup instructions
  - Calibration process step-by-step
  - Troubleshooting common issues
  - Advanced calibration options
  - Verification procedures

## 📋 Quick Reference

### Project Files

- **[PROJECT_SUMMARY.md](../PROJECT_SUMMARY.md)** - Executive summary of the framework
- **[requirements.txt](../requirements.txt)** - Python dependencies
- **[setup.py](../setup.py)** - Package configuration
- **[configs/default.yaml](../configs/default.yaml)** - Configuration template

### Checklists

- **[IMPLEMENTATION_CHECKLIST.md](../IMPLEMENTATION_CHECKLIST.md)** - Detailed task list for implementation

## 📊 Documentation Map

```
docs/
│
├── 🌟 GETTING_STARTED.md      # START HERE - Implementation roadmap
│
├── 🏗️  ARCHITECTURE.md         # System design & data flow
│
├── 📘 MODULE_SPECS.md          # Complete API specifications
│
├── 🔌 OUTPUT_PROTOCOLS.md      # Communication protocols
│
├── 👨‍💻 DEVELOPMENT.md           # Development guidelines
│
├── 🎯 CALIBRATION_GUIDE.md     # User calibration guide
│
└── 📑 INDEX.md                 # This file
```

## 🎯 By Task

### I want to

**Understand the system:**
→ Read [ARCHITECTURE.md](ARCHITECTURE.md)

**Implement a module:**
→ Check [MODULE_SPECS.md](MODULE_SPECS.md) for the module's API  
→ Follow guidelines in [DEVELOPMENT.md](DEVELOPMENT.md)

**Add a new output protocol:**
→ See examples in [OUTPUT_PROTOCOLS.md](OUTPUT_PROTOCOLS.md)  
→ Follow pattern in [DEVELOPMENT.md](DEVELOPMENT.md#adding-new-features)

**Set up cameras:**
→ Follow [CALIBRATION_GUIDE.md](CALIBRATION_GUIDE.md)

**Run tests:**
→ See testing section in [DEVELOPMENT.md](DEVELOPMENT.md#testing)

**Track progress:**
→ Use [IMPLEMENTATION_CHECKLIST.md](../IMPLEMENTATION_CHECKLIST.md)

**Configure the system:**
→ Edit [configs/default.yaml](../configs/default.yaml)  
→ Reference comments in the file

## 📈 Documentation Stats

- **Total Documentation**: ~137KB
- **Number of Documents**: 7 major files
- **Modules Documented**: 22
- **Code Examples**: 50+
- **Diagrams**: 10+

## 🔍 Search Tips

### Finding Information

**Architecture questions:**

- Component interactions → ARCHITECTURE.md
- Threading model → ARCHITECTURE.md
- Error handling → ARCHITECTURE.md

**Implementation questions:**

- Class interface → MODULE_SPECS.md
- Method signature → MODULE_SPECS.md
- Data structures → MODULE_SPECS.md

**Protocol questions:**

- Packet format → OUTPUT_PROTOCOLS.md
- Latency comparison → OUTPUT_PROTOCOLS.md
- Integration examples → OUTPUT_PROTOCOLS.md

**Development questions:**

- Code style → DEVELOPMENT.md
- Testing approach → DEVELOPMENT.md
- How to add features → DEVELOPMENT.md

**User questions:**

- Setup process → CALIBRATION_GUIDE.md
- Troubleshooting → CALIBRATION_GUIDE.md
- Configuration → configs/default.yaml

## 📚 Reading Order

### For Developers (Recommended)

1. **[README.md](../README.md)** - Get the big picture (10 min)
2. **[ARCHITECTURE.md](ARCHITECTURE.md)** - Understand system design (30 min)
3. **[MODULE_SPECS.md](MODULE_SPECS.md)** - Study APIs (1 hour, reference as needed)
4. **[DEVELOPMENT.md](DEVELOPMENT.md)** - Learn development practices (30 min)
5. **[GETTING_STARTED.md](GETTING_STARTED.md)** - Plan implementation (20 min)

### For Users

1. **[README.md](../README.md)** - Project overview (10 min)
2. **[CALIBRATION_GUIDE.md](CALIBRATION_GUIDE.md)** - Setup and calibration (30 min)
3. **[configs/default.yaml](../configs/default.yaml)** - Configuration options (15 min)

### For AI Assistants

When providing implementation assistance, reference:

1. **[MODULE_SPECS.md](MODULE_SPECS.md)** - For exact interfaces
2. **[DEVELOPMENT.md](DEVELOPMENT.md)** - For code style and patterns
3. **[ARCHITECTURE.md](ARCHITECTURE.md)** - For system context

## 🎓 Learning Path

### Week 1: Understanding

- [ ] Read README and overview docs
- [ ] Study architecture
- [ ] Review module specifications

### Week 2-3: Foundation

- [ ] Implement utilities
- [ ] Set up configuration
- [ ] Write first tests

### Week 4-6: Vision

- [ ] Camera integration
- [ ] Vision processing
- [ ] Detection algorithms

### Week 7-8: Calibration

- [ ] Calibration UI
- [ ] Transform calculations
- [ ] Data persistence

### Week 9-10: Output

- [ ] Output adapters
- [ ] Protocol implementation
- [ ] Integration testing

### Week 11-12: Polish

- [ ] Optimization
- [ ] Documentation updates
- [ ] Release preparation

## 💡 Tips

### For Effective Use

1. **Bookmark this index** - Quick access to all docs
2. **Use Ctrl+F** - Search within documents
3. **Check MODULE_SPECS.md** - Before implementing any module
4. **Refer to DEVELOPMENT.md** - For coding standards
5. **Update CHECKLIST** - Track your progress

### For AI Assistance

When asking AI for help, provide context:

```
"According to MODULE_SPECS.md, the HandDetector class should..."
"Following DEVELOPMENT.md guidelines, implement..."
"Based on ARCHITECTURE.md's data flow, the pipeline should..."
```

## 🔗 External Resources

### Computer Vision

- [OpenCV Documentation](https://docs.opencv.org/)
- [Camera Calibration Guide](https://docs.opencv.org/master/dc/dbb/tutorial_py_calibration.html)

### Python

- [Python Type Hints - PEP 484](https://www.python.org/dev/peps/pep-0484/)
- [pytest Documentation](https://docs.pytest.org/)

### Hardware

- Linux uinput documentation (for HID/keyboard)
- UDP socket programming tutorials

## 📞 Quick Commands

```bash
# View README
cat README.md

# Search all docs for a term
grep -r "search_term" docs/

# List all markdown files
find docs/ -name "*.md"

# Count total documentation size
find docs/ -name "*.md" -exec wc -c {} + | tail -1
```

## ✅ Document Checklist

Track which documents you've read:

- [ ] README.md
- [ ] FRAMEWORK_OVERVIEW.md
- [ ] ARCHITECTURE.md
- [ ] MODULE_SPECS.md
- [ ] OUTPUT_PROTOCOLS.md
- [ ] DEVELOPMENT.md
- [ ] CALIBRATION_GUIDE.md
- [ ] GETTING_STARTED.md
- [ ] PROJECT_SUMMARY.md

---

**Documentation Version**: 1.0.0  
**Last Updated**: 2026-01-10  
**Framework Status**: Ready for Implementation ✅
